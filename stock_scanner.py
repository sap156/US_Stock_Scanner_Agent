import os
#!/usr/bin/env python3
"""
Multi-Agent Stock Scanner using LangGraph
A team of AI agents working together to scan and analyze stocks
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import time
import warnings
from typing import TypedDict, List, Dict, Any
from dataclasses import dataclass
import json
import sys

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

warnings.filterwarnings('ignore')

# State definition for the multi-agent workflow
class StockScannerState(TypedDict):
    symbols: List[str]
    market_name: str
    raw_data: Dict[str, Any]
    filtered_stocks: List[Dict[str, Any]]
    technical_analysis: Dict[str, Any]
    final_results: List[Dict[str, Any]]  # Use list, not DataFrame
    scan_metadata: Dict[str, Any]
    agent_reports: Dict[str, str]
    current_step: str

@dataclass
class AgentConfig:
    name: str
    role: str
    max_workers: int = 5

class BaseAgent:
    """Base class for all agents in the stock scanning team"""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.name = config.name
        self.role = config.role
        
    def log(self, message: str):
        print(f"[{self.name.upper()}] {message}")



# New agent: MarketCapFilterAgent
class MarketCapFilterAgent(BaseAgent):
    """Agent responsible for filtering stocks by market cap greater than 5B USD"""
    def __init__(self):
        super().__init__(AgentConfig(
            name="MarketCapFilter", 
            role="Filters stocks by market cap greater than 5B USD"
        ))
    def get_symbols_from_csv(self, csv_file: str) -> list:
        import pandas as pd
        self.log(f"Step: Reading symbols from {csv_file}")
        df = pd.read_csv(csv_file)
        # Assume the symbol column is named 'Symbol' (case-insensitive)
        symbol_col = None
        for col in df.columns:
            if col.strip().lower() == 'symbol':
                symbol_col = col
                break
        if symbol_col is None:
            raise ValueError("CSV does not contain a 'Symbol' column")
        symbols = df[symbol_col].dropna().astype(str).tolist()
        self.log(f"Step: Loaded {len(symbols)} symbols from CSV")
        return symbols

    def execute(self, state: StockScannerState) -> StockScannerState:
        self.log("Step: Executing MarketCapFilterAgent (CSV mode)")
        # Use the provided CSV file for symbols
        csv_file = "nasdaq_screener.csv"
        symbols = self.get_symbols_from_csv(csv_file)
        state['symbols'] = symbols
        state['agent_reports']['market_cap_filter'] = f"Loaded {len(symbols)} stocks from CSV with >$5B USD market cap"
        state['current_step'] = 'market_cap_filtered'
        self.log("Step: MarketCapFilterAgent complete (CSV mode)")
        return state

# Updated DataCollectionAgent: only fetches data for already filtered symbols
class DataCollectionAgent(BaseAgent):
    """Agent responsible for collecting stock data from Yahoo Finance"""
    def __init__(self):
        super().__init__(AgentConfig(
            name="DataCollector", 
            role="Fetches historical stock data and basic company information"
        ))
    def calculate_ema(self, data, period):
        return data.ewm(span=period).mean()
    def collect_stock_data(self, symbol: str, period: str = "1y") -> Dict[str, Any]:
        self.log(f"Step: Collecting data for {symbol}")
        try:
            hist = yf.download(symbol, period=period, progress=False)
            if hist.empty or len(hist) < 50:
                self.log(f"Step: Not enough data for {symbol}")
                return None
            hist['EMA_21'] = self.calculate_ema(hist['Close'], 21)
            hist['EMA_50'] = self.calculate_ema(hist['Close'], 50)
            hist['Volume_EMA_50'] = self.calculate_ema(hist['Volume'], 50)
            hist['High_Max_250'] = hist['High'].rolling(window=min(250, len(hist))).max()
            hist['Close_Min_260'] = hist['Close'].rolling(window=min(260, len(hist))).min()
            hist['Pct_Change'] = hist['Close'].pct_change() * 100
            latest_data = hist.iloc[-1]
            # Ensure all values are scalars, not Series
            def to_scalar(val):
                if isinstance(val, pd.Series):
                    if len(val) == 1:
                        return val.iloc[0]
                    else:
                        return val.item() if hasattr(val, 'item') and val.size == 1 else val
                return val
            try:
                stock = yf.Ticker(symbol)
                info = stock.info
                company_name = info.get('longName', symbol)
                market_cap = info.get('marketCap', 0)
                sector = info.get('sector', 'Unknown')
            except:
                company_name = symbol
                market_cap = 0
                sector = 'Unknown'
            collected = {
                'symbol': symbol,
                'company_name': company_name,
                'sector': sector,
                'market_cap': to_scalar(market_cap),
                'current_price': to_scalar(latest_data['Close']),
                'pct_change': to_scalar(latest_data['Pct_Change']),
                'volume': to_scalar(latest_data['Volume']),
                'ema_21': to_scalar(latest_data['EMA_21']),
                'ema_50': to_scalar(latest_data['EMA_50']),
                'volume_ema_50': to_scalar(latest_data['Volume_EMA_50']),
                'high_max_250': to_scalar(latest_data['High_Max_250']),
                'close_min_260': to_scalar(latest_data['Close_Min_260']),
                'data_quality': 'good'
            }
            self.log(f"Step: Data collected for {symbol}: {collected}")
            return collected
        except Exception as e:
            self.log(f"Step: Error collecting data for {symbol}: {e}")
            return None
    def execute(self, state: StockScannerState) -> StockScannerState:
        self.log("Step: Executing DataCollectionAgent")
        self.log(f"Step: Starting data collection for {len(state['symbols'])} symbols in {state['market_name']}")
        raw_data = {}
        successful_collections = 0
        for symbol in state['symbols']:
            data = self.collect_stock_data(symbol)
            if data:
                raw_data[symbol] = data
                successful_collections += 1
        self.log(f"Step: Successfully collected data for {successful_collections}/{len(state['symbols'])} symbols")
        state['raw_data'] = raw_data
        state['agent_reports']['data_collector'] = f"Collected data for {successful_collections} stocks"
        state['current_step'] = 'data_collection_complete'
        self.log("Step: DataCollectionAgent complete")
        return state

class TechnicalAnalysisAgent(BaseAgent):
    """Agent responsible for technical analysis and filtering"""
    
    def __init__(self):
        super().__init__(AgentConfig(
            name="TechnicalAnalyst", 
            role="Applies technical filters and analyzes market conditions"
        ))
        
    def apply_technical_filters(self, stock_data: Dict[str, Any]) -> tuple[bool, str]:
        self.log(f"Step: Applying technical filters to {stock_data.get('symbol', 'UNKNOWN')}")
        """Apply only the specified technical filters"""
        if not stock_data:
            return False, "No data available"
        try:
            # Ensure all values are scalars, not Series
            def to_scalar(val):
                if isinstance(val, pd.Series):
                    if len(val) == 1:
                        return val.iloc[0]
                    else:
                        return val.item() if hasattr(val, 'item') and val.size == 1 else val
                return val

            current_price = to_scalar(stock_data['current_price'])
            market_cap = to_scalar(stock_data.get('market_cap', 0))
            pct_change = to_scalar(stock_data['pct_change'])
            high_max_250 = to_scalar(stock_data['high_max_250'])
            ema_50 = to_scalar(stock_data['ema_50'])
            ema_21 = to_scalar(stock_data['ema_21'])
            volume_ema_50 = to_scalar(stock_data['volume_ema_50'])
            close_min_260 = to_scalar(stock_data['close_min_260'])
            reasons = []
            # Check for NaN values
            if pd.isna(current_price) or pd.isna(pct_change):
                return False, "Invalid price data"
            # Filter 1: Small and Mid Cap (market cap < $10B)
            if pd.isna(market_cap) or market_cap >= 1e10:
                reasons.append("Market cap not small/mid (<$10B)")
            # Filter 2: Price >= 75% of 250-day high
            if pd.isna(high_max_250) or current_price < (high_max_250 * 0.75):
                reasons.append("Below 75% of 250-day high")
            # Filter 3: Price >= $20
            if current_price < 20:
                reasons.append("Price below $20")
            # Filter 4: Price >= EMA(50)
            if pd.isna(ema_50) or current_price < ema_50:
                reasons.append("Below 50-day EMA")
            # Filter 5: Price >= EMA(21)
            if pd.isna(ema_21) or current_price < ema_21:
                reasons.append("Below 21-day EMA")
            # Filter 6: Daily % change between -3% and 5%
            if pct_change > 5:
                reasons.append("Daily change > 5%")
            if pct_change <= -3:
                reasons.append("Daily change <= -3%")
            # Filter 7: 50-day average dollar volume >= $75M
            if pd.isna(volume_ema_50) or (volume_ema_50 * current_price) < 75000000:
                reasons.append("Insufficient liquidity (50-day avg dollar vol < $75M)")
            # Filter 8: Price >= 2x 260-day low
            if pd.isna(close_min_260) or current_price < (close_min_260 * 2):
                reasons.append("Below 2x 260-day low")
            if reasons:
                return False, "; ".join(reasons)
            return True, "Passed all filters"
        except Exception as e:
            return False, f"Filter error: {e}"
    
    def analyze_market_strength(self, filtered_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze overall market conditions from filtered stocks"""
        if not filtered_data:
            return {"strength": "weak", "trend": "neutral", "analysis": "No stocks passed filters"}
        
        price_changes = [data['pct_change'] for data in filtered_data.values()]
        avg_change = np.mean(price_changes)
        
        strength = "strong" if len(filtered_data) > 10 else "moderate" if len(filtered_data) > 5 else "weak"
        trend = "bullish" if avg_change > 1 else "bearish" if avg_change < -1 else "neutral"
        
        return {
            "strength": strength,
            "trend": trend,
            "avg_change": round(avg_change, 2),
            "stocks_passed": len(filtered_data),
            "analysis": f"Market showing {strength} performance with {trend} trend"
        }
    
    def execute(self, state: StockScannerState) -> StockScannerState:
        self.log("Step: Executing TechnicalAnalysisAgent")
        self.log("Step: Starting technical analysis and filtering")
        filtered_stocks = {}
        filter_results = {}
        for symbol, data in state['raw_data'].items():
            passed, reason = self.apply_technical_filters(data)
            filter_results[symbol] = {"passed": passed, "reason": reason}
            if passed:
                filtered_stocks[symbol] = data
                self.log(f"Step: ✓ {symbol} passed all filters")
            else:
                self.log(f"Step: ✗ {symbol} filtered out: {reason}")
        technical_analysis = self.analyze_market_strength(filtered_stocks)
        self.log(f"Step: Analysis complete: {len(filtered_stocks)} stocks passed filters")
        state['filtered_stocks'] = list(filtered_stocks.values())
        state['technical_analysis'] = technical_analysis
        state['agent_reports']['technical_analyst'] = f"Filtered to {len(filtered_stocks)} qualifying stocks"
        state['current_step'] = 'technical_analysis_complete'
        self.log("Step: TechnicalAnalysisAgent complete")
        return state

class ReportingAgent(BaseAgent):
    def generate_screening_checklist(self, stock: dict) -> str:
        """Generate a screening criteria checklist for a single stock."""
        # Extract values
        name = stock.get('company_name', '')
        symbol = stock.get('symbol', '')
        market_cap = stock.get('market_cap', 0)
        price = stock.get('current_price', 0)
        high_max_250 = stock.get('high_max_250', 0)
        ema_50 = stock.get('ema_50', 0)
        ema_21 = stock.get('ema_21', 0)
        pct_change = stock.get('pct_change', 0)
        volume_ema_50 = stock.get('volume_ema_50', 0)
        close_min_260 = stock.get('close_min_260', 0)
        # Calculated values
        meets = lambda cond: '✔️' if cond else '❌'
        # Market Cap: $300M–$10B
        mc_req = "$300M–$10B"
        mc_val = f"${market_cap/1e9:.1f}B" if market_cap else "N/A"
        mc_meets = meets(3e8 <= market_cap < 1e10)
        # Price ≥ 75% of 250-day high
        p75 = high_max_250 * 0.75 if high_max_250 else 0
        p75_req = f"≥${p75:.2f} (75% of ${high_max_250:.2f})" if high_max_250 else "N/A"
        p75_meets = meets(price >= p75 and high_max_250)
        # Price ≥ $20
        p20_meets = meets(price >= 20)
        # Price ≥ 50-day EMA
        ema50_meets = meets(price >= ema_50 and ema_50)
        # Price ≥ 21-day EMA
        ema21_meets = meets(price >= ema_21 and ema_21)
        # Daily % change between -3% and 5%
        pct_req = "-3% < Change ≤ 5%"
        pct_meets = meets(-3 < pct_change <= 5)
        # 50-day avg dollar volume ≥ $75M
        vol_req = "$75,000,000"
        vol_val = f"~${volume_ema_50*price:,.0f}" if volume_ema_50 and price else "N/A"
        vol_meets = meets(volume_ema_50 and (volume_ema_50*price) >= 75000000)
        # Price ≥ 2x 260-day low
        low2x = close_min_260 * 2 if close_min_260 else 0
        low2x_req = f"≥${low2x:.2f} (2 × ${close_min_260:.2f})" if close_min_260 else "N/A"
        low2x_meets = meets(price >= low2x and close_min_260)
        # Build checklist table
        checklist = [
            ["Criteria", "Requirement", f"{symbol} Value", "Meets?"],
            ["Market Cap (Small/Mid)", mc_req, mc_val, mc_meets],
            ["Current Price ≥ 75% of 250-day High", p75_req, f"${price:.2f}", p75_meets],
            ["Current Price ≥ $20", "$20", f"${price:.2f}", p20_meets],
            ["Current Price ≥ 50-day EMA", f"${ema_50:.2f}", f"${price:.2f}", ema50_meets],
            ["Current Price ≥ 21-day EMA", f"${ema_21:.2f}", f"${price:.2f}", ema21_meets],
            ["Daily % Change", pct_req, f"{pct_change:.2f}%", pct_meets],
            ["50-day Avg Dollar Volume ≥ $75M", vol_req, vol_val, vol_meets],
            ["Current Price ≥ 2x 260-day Low", low2x_req, f"${price:.2f}", low2x_meets],
        ]
        # Format as markdown table
        lines = ["| " + " | ".join(row) + " |" for row in checklist]
        lines.insert(1, "|---|---|---|---|")
        return f"\nScreening Criteria Checklist for {name} ({symbol}):\n" + "\n".join(lines)
    """Agent responsible for generating final reports and outputs"""
    
    def __init__(self):
        super().__init__(AgentConfig(
            name="Reporter", 
            role="Generates final reports and CSV outputs"
        ))
    
    def create_detailed_report(self, state: StockScannerState) -> str:
        """Generate a comprehensive analysis report"""
        report = []
        report.append("="*80)
        report.append("MULTI-AGENT STOCK SCANNER REPORT")
        report.append("="*80)
        # Scan metadata
        metadata = state['scan_metadata']
        duration = metadata.get('duration', None)
        if duration is None:
            duration_str = 'N/A'
        else:
            duration_str = f"{duration:.2f} seconds"
        report.append(f"Scan Date: {metadata.get('timestamp', 'N/A')}")
        report.append(f"Market: {state.get('market_name', 'N/A')}")
        report.append(f"Total Symbols Analyzed: {metadata.get('total_symbols', 'N/A')}")
        report.append(f"Scan Duration: {duration_str}")
        report.append("")
        # Agent reports
        report.append("AGENT ACTIVITY SUMMARY:")
        for agent, activity in state['agent_reports'].items():
            report.append(f"• {agent.title()}: {activity}")
        report.append("")
        # Technical analysis
        tech_analysis = state['technical_analysis']
        report.append("MARKET ANALYSIS:")
        report.append(f"• Market Strength: {tech_analysis.get('strength', 'N/A').title() if tech_analysis.get('strength') else 'N/A'}")
        report.append(f"• Trend Direction: {tech_analysis.get('trend', 'N/A').title() if tech_analysis.get('trend') else 'N/A'}")
        report.append(f"• Average Change: {tech_analysis.get('avg_change', 'N/A')}%")
        report.append(f"• Stocks Qualified: {tech_analysis.get('stocks_passed', 'N/A')}")
        report.append("")
        # Filter criteria
        report.append("APPLIED FILTERS:")
        filters = [
            "1. Price ≥ 75% of 250-day high",
            "2. Price ≥ $20",
            "3. Price ≥ 50-day EMA", 
            "4. Price ≥ 21-day EMA",
            "5. Daily change ≤ 5%",
            "6. Daily change > -3%",
            "7. Liquidity: Volume×Price ≥ $75M",
            "8. Price ≥ 2× 260-day low"
        ]
        for filter_desc in filters:
            report.append(f"• {filter_desc}")
        return "\n".join(report)
    
    def execute(self, state: StockScannerState) -> StockScannerState:
        self.log("Step: Executing ReportingAgent")
        self.log("Step: Generating final reports and outputs")
        # Create DataFrame from filtered stocks
        if state['filtered_stocks']:
            results_data = []
            for stock in state['filtered_stocks']:
                # Convert all values to native Python types
                def to_native(val):
                    if isinstance(val, (np.generic, np.ndarray)):
                        return val.item() if hasattr(val, 'item') else val.tolist()
                    return val
                results_data.append({
                    'StockName': str(stock['company_name']),
                    'Symbol': str(stock['symbol']),
                    '% Chg': float(round(float(stock['pct_change']), 2)),
                    'Price': float(round(float(stock['current_price']), 2)),
                    'Volume': int(stock['volume']),
                    'Sector': str(stock['sector'])
                })
            final_df = pd.DataFrame(results_data)
            final_df = final_df.sort_values(by='% Chg', ascending=False)
        else:
            final_df = pd.DataFrame(columns=['StockName', 'Symbol', '% Chg', 'Price', 'Volume', 'Sector'])
        # Generate detailed report
        detailed_report = self.create_detailed_report(state)
        # Add screening checklist for each final result
        if state['filtered_stocks']:
            for stock in state['filtered_stocks']:
                detailed_report += "\n\n" + self.generate_screening_checklist(stock)
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_filename = f"multi_agent_stock_scan_{timestamp}.csv"
        report_filename = f"scan_report_{timestamp}.txt"
        # Save CSV
        final_df.to_csv(csv_filename, index=False)
        # Save detailed report
        with open(report_filename, 'w') as f:
            f.write(detailed_report)
        self.log(f"Step: Reports saved: {csv_filename}, {report_filename}")
        # Store results as a list of dicts to avoid msgpack serialization issues with DataFrame and numpy types
        def convert_dict_native(d):
            return {k: (v.item() if isinstance(v, (np.generic, np.ndarray)) else v) for k, v in d.items()}
        state['final_results'] = [convert_dict_native(row) for row in final_df.to_dict(orient='records')] if not final_df.empty else []
        state['agent_reports']['reporter'] = f"Generated {len(final_df)} final results"
        state['current_step'] = 'reporting_complete'
        self.log("Step: ReportingAgent complete")
        return state

class CoordinatorAgent(BaseAgent):
    """Agent responsible for coordinating the entire workflow"""
    
    def __init__(self):
        super().__init__(AgentConfig(
            name="Coordinator", 
            role="Orchestrates the multi-agent stock scanning workflow"
        ))
        
        # Initialize other agents
        self.data_collector = DataCollectionAgent()
        self.technical_analyst = TechnicalAnalysisAgent()
        self.reporter = ReportingAgent()
        
    def initialize_scan(self, state: StockScannerState) -> StockScannerState:
        self.log("Step: Initializing multi-agent stock scan")
        # Setup initial state
        start_time = time.time()
        state['scan_metadata'] = {
            'start_time': start_time,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_symbols': len(state['symbols'])
        }
        state['agent_reports'] = {}
        state['current_step'] = 'initialized'
        self.log(f"Step: Scan initialized for {len(state['symbols'])} symbols in {state['market_name']}")
        self.log("Step: CoordinatorAgent initialize_scan complete")
        return state
    
    def finalize_scan(self, state: StockScannerState) -> StockScannerState:
        self.log("Step: Finalizing multi-agent stock scan")
        end_time = time.time()
        duration = end_time - state['scan_metadata']['start_time']
        state['scan_metadata']['duration'] = duration
        state['scan_metadata']['end_time'] = end_time
        state['current_step'] = 'completed'
        self.log(f"Step: Multi-agent scan completed in {duration:.2f} seconds")
        # Print summary
        print("\n" + "="*80)
        print("MULTI-AGENT SCAN SUMMARY")
        print("="*80)
        print(f"Stocks Found: {len(state['final_results'])}")
        print(f"Market Analyzed: {state['market_name']}")
        print(f"Duration: {duration:.2f} seconds")
        print("="*80)
        if len(state['final_results']) > 0:
            print("\nTop Results:")
            # Print as table if possible
            import pandas as pd
            try:
                df = pd.DataFrame(state['final_results'])
                print(df.head(10).to_string(index=False))
            except Exception:
                print(state['final_results'][:10])
        self.log("Step: CoordinatorAgent finalize_scan complete")
        return state

def create_multi_agent_graph():
    """Create the LangGraph workflow for multi-agent stock scanning"""
    workflow = StateGraph(StockScannerState)
    coordinator = CoordinatorAgent()
    # Add new agents
    market_cap_filter_agent = MarketCapFilterAgent()
    # Add nodes for each agent
    workflow.add_node("initialize", coordinator.initialize_scan)
    workflow.add_node("market_cap_filter", market_cap_filter_agent.execute)
    workflow.add_node("data_collection", coordinator.data_collector.execute)
    workflow.add_node("technical_analysis", coordinator.technical_analyst.execute)
    workflow.add_node("reporting", coordinator.reporter.execute)
    workflow.add_node("finalize", coordinator.finalize_scan)
    # Define the workflow edges
    workflow.set_entry_point("initialize")
    workflow.add_edge("initialize", "market_cap_filter")
    workflow.add_edge("market_cap_filter", "data_collection")
    workflow.add_edge("data_collection", "technical_analysis")
    workflow.add_edge("technical_analysis", "reporting")
    workflow.add_edge("reporting", "finalize")
    workflow.add_edge("finalize", END)
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    return app



def main():
    """Main function to run the multi-agent stock scanner"""
    print("🤖 MULTI-AGENT STOCK SCANNER")
    print("="*60)
    print("Team of AI Agents:")
    print("• DataCollector: Fetches stock data")
    print("• TechnicalAnalyst: Applies filters")
    print("• Reporter: Generates outputs")
    print("• Coordinator: Orchestrates workflow")
    print("="*60)
    
    # Create the multi-agent workflow
    app = create_multi_agent_graph()
    
    # Prepare initial state for US market (symbols will be fetched by SymbolUniverseAgent)
    initial_state = StockScannerState(
        symbols=[],
        market_name="US Market",
        raw_data={},
        filtered_stocks=[],
        technical_analysis={},
        final_results=[],  # Use list, not DataFrame
        scan_metadata={},
        agent_reports={},
        current_step="starting"
    )
    
    # Execute the multi-agent workflow
    config = {"configurable": {"thread_id": f"scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}"}}
    
    try:
        # Run the workflow
        final_state = app.invoke(initial_state, config=config)
        
        print(f"\n✅ Multi-agent scan completed successfully!")
        print(f"📊 Results: {len(final_state['final_results'])} qualifying stocks found")
        
    except Exception as e:
        print(f"❌ Error during multi-agent scan: {e}")
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Scan interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"💥 Critical error: {e}")
        sys.exit(1)