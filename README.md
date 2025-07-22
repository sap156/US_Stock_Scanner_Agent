# Multi-Agent US Stock Scanner

This project is a robust, multi-agent stock scanner for the US equity market, leveraging the LangGraph framework to coordinate a team of AI agents. The scanner is designed to automate the process of filtering, analyzing, and reporting on US stocks, focusing on small and mid cap companies with strong technical and liquidity characteristics.

## Features
- **Multi-Agent Architecture:** Modular agents for data collection, filtering, technical analysis, and reporting.
- **Automated Symbol Universe:** Reads a CSV of US stocks (e.g., from NASDAQ/NYSE/AMEX) for analysis.
- **Market Cap Filtering:** Focuses on small and mid cap stocks (market cap < $10B).
- **Technical Filters:**
  - Price ≥ 75% of 250-day high
  - Price ≥ $20
  - Price ≥ 50-day EMA
  - Price ≥ 21-day EMA
  - Daily % change between -3% and 5%
  - 50-day average dollar volume ≥ $75M
  - Price ≥ 2× 260-day low
- **Batch Data Fetching:** Efficiently fetches historical and fundamental data using yfinance.
- **Progress Logging:** Detailed logs and progress at each step.
- **Comprehensive Reporting:** Generates CSV and text reports with summary statistics and filter criteria.

## Project Structure

```
US_Stock_Scanner_Agent_clean/
├── stock_scanner.py         # Main multi-agent scanner script
├── requirements.txt         # Python dependencies
├── nasdaq_screener.csv      # (User-provided) List of US stock symbols to scan
└── ... (output files: CSV, TXT reports)
```

## Setup Instructions

1. **Clone the repository and navigate to the project directory:**
   ```sh
   git clone <your-repo-url>
   cd US_Stock_Scanner_Agent_clean
   ```

2. **(Recommended) Create a virtual environment:**
   ```sh
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

4. **Prepare your symbol universe:**
   - Place a CSV file named `nasdaq_screener.csv` in the project directory.
   - The CSV must have a column named `Symbol` (case-insensitive) listing the stock tickers to scan.

5. **Run the scanner:**
   ```sh
   python stock_scanner.py
   ```

## Output
- **CSV Report:** `multi_agent_stock_scan_<timestamp>.csv` — Filtered stocks with key data.
- **Text Report:** `scan_report_<timestamp>.txt` — Detailed summary and filter criteria.

## Agent Workflow
1. **CoordinatorAgent:** Orchestrates the workflow.
2. **MarketCapFilterAgent:** Loads symbols from CSV (assumed already filtered for market cap > $5B).
3. **DataCollectionAgent:** Fetches historical and fundamental data for each symbol.
4. **TechnicalAnalysisAgent:** Applies technical and liquidity filters.
5. **ReportingAgent:** Generates CSV and text reports.

## Customization
- To change filter thresholds or add new filters, edit the `TechnicalAnalysisAgent` class in `stock_scanner.py`.
- To use a different symbol universe, replace `nasdaq_screener.csv` with your own CSV file.

## Dependencies
- Python 3.9+
- yfinance
- pandas
- numpy
- langgraph

Install all dependencies with:
```sh
pip install -r requirements.txt
```

## Notes
- The scanner is designed for US stocks only. For other markets, additional logic is required.
- The workflow is modular and can be extended with new agents or analysis steps.
- Ensure your CSV file is up to date for best results.

## License
MIT License
