# Streamlit Temp - Options Chain Analysis

A comprehensive options chain analysis tool that combines data processing, visualization, and AI-powered image analysis.

## Files Overview

### 📊 Data & Analysis Files
- **`options.csv`** - Options chain data with contract details, volume, implied volatility, etc.
- **`temp.py`** - Main analysis script using Florence-2 for image analysis and comprehensive options data processing
- **`temp_chart_creator.py`** - Creates detailed options chain visualization charts
- **`backup.py`** - Streamlit web application for advanced stock market dashboard

### 📈 Generated Files
- **`options_chain_analysis.png`** - Visual representation of options chain data

## Features

### 🔍 Options Analysis
- **Volume Analysis**: Call vs Put volume ratios
- **Strike Analysis**: Volume distribution across strike prices  
- **Expiration Analysis**: Activity across different expiration dates
- **Implied Volatility**: IV patterns and volatility smile analysis
- **Moneyness**: ITM/OTM contract analysis

### 📊 Visualizations
- Call/Put volume by strike price
- Implied volatility smile charts
- Put-Call ratio analysis
- Options chain heatmaps

### 🤖 AI-Powered Analysis
- Florence-2 vision model integration for chart analysis
- Contextual analysis combining visual and data insights
- Automated pattern recognition in options charts

### 🌐 Web Interface (backup.py)
- Real-time stock data fetching
- Interactive options chain visualization
- Advanced charting with technical indicators
- Database integration for data persistence

## Usage

### 1. Generate Options Charts
```bash
python temp_chart_creator.py
```
Creates comprehensive options chain visualizations.

### 2. Run Analysis
```bash
python temp.py
```
Performs detailed analysis of options data and any chart images in the directory.

### 3. Launch Web Dashboard
```bash
streamlit run backup.py
```
Starts the interactive web application.

## Key Insights from Current Data (BMNR)

- **Total Volume**: 92,337 contracts
- **Put-Call Ratio**: 0.41 (Bullish sentiment)
- **Current Stock Price**: $45.57
- **High Activity Strikes**: $48-50 (calls), $44-45 (puts)
- **Primary Expiration**: 2025-08-29 (near-term event-driven activity)

## Technical Requirements

- Python 3.9+
- Required packages: `pandas`, `matplotlib`, `seaborn`, `PIL`, `transformers`, `torch`
- For web app: `streamlit`, `plotly`, `yfinance`

## Data Structure

The options.csv contains:
- Contract symbols and details
- Strike prices and expiration dates
- Volume and open interest
- Bid/Ask spreads
- Implied volatility
- Current stock price context

## Analysis Methodology

1. **Volume-Based Analysis**: Identifies high-activity strikes and expiration dates
2. **Sentiment Analysis**: Uses put-call ratios and positioning
3. **Visual Pattern Recognition**: AI analysis of chart patterns
4. **Risk Assessment**: Implied volatility and moneyness analysis

---

*Last Updated: August 28, 2025*
