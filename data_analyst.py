"""
Financial Data Analysis Script
Analyzing cryptocurrency market data with technical indicators

This script demonstrates data ingestion from CSV, data wrangling,
custom analysis functions, and visualization capabilities.
"""

# Standard library imports
import warnings
from datetime import datetime
from typing import Dict

# Third-party libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Try to use seaborn for better plots, but it's optional
try:
    import seaborn as sns
    sns.set_style("whitegrid")
    USE_SEABORN = True
except ImportError:
    USE_SEABORN = False
    print("Note: seaborn not available, using default matplotlib style")

# Configure plot settings
plt.rcParams['figure.figsize'] = (12, 6)
plt.style.use('default')


def ingest_data_from_csv(filepath: str) -> pd.DataFrame:
    """
    Load and process CSV data file.
    Handles different data types and cleans the dataset.
    """
    # Read CSV file - using Date column as index for time series
    df = pd.read_csv(filepath, index_col="Date", parse_dates=True)
    
    # Need to handle different data types properly
    # Price and volume columns should be numeric (float)
    numeric_cols = []
    for col in df.columns:
        if 'Close' in col or 'Volume' in col or 'High' in col or 'Low' in col:
            numeric_cols.append(col)
    
    # Convert to float, handling any non-numeric values
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)
    
    # Fix timezone issues if present (some APIs include timezone info)
    if hasattr(df.index, 'tz') and df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    
    # Clean up: remove duplicate dates and fill missing values
    df = df[~df.index.duplicated(keep='last')]
    df = df.sort_index()
    df = df.ffill()  # Forward fill for missing prices
    
    # Print summary
    print(f"✓ Loaded {len(df)} rows from {filepath}")
    print(f"  Date range: {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}")
    print(f"  Columns: {', '.join(df.columns[:3])}..." if len(df.columns) > 3 else f"  Columns: {', '.join(df.columns)}")
    
    return df


def calculate_technical_indicators(df: pd.DataFrame, price_col: str = "BTCUSDT_Close") -> pd.DataFrame:
    """
    My custom function to calculate various technical indicators.
    I use RSI, MACD, moving averages, and Bollinger Bands for analysis.
    """
    df = df.copy()  # Don't modify original
    prices = df[price_col]
    
    # RSI calculation - standard 14-period RSI
    # RSI measures momentum, values above 70 = overbought, below 30 = oversold
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    # Avoid division by zero
    rs = gain / loss.replace(0, 1e-10)
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD - popular trend-following indicator
    ema12 = prices.ewm(span=12, adjust=False).mean()  # Fast EMA
    ema26 = prices.ewm(span=26, adjust=False).mean()    # Slow EMA
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()  # Signal line
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']  # Histogram
    
    # Simple moving averages - helps identify trends
    df['MA_20'] = prices.rolling(window=20).mean()  # Short-term
    df['MA_50'] = prices.rolling(window=50).mean()  # Medium-term
    
    # Bollinger Bands - volatility indicator
    # When price touches upper band, might be overbought; lower band = oversold
    df['BB_Mid'] = prices.rolling(window=20).mean()
    bb_std = prices.rolling(window=20).std()
    df['BB_Upper'] = df['BB_Mid'] + 2 * bb_std
    df['BB_Lower'] = df['BB_Mid'] - 2 * bb_std
    
    # ATR (Average True Range) - measures volatility
    # Try to use high/low if available, otherwise approximate with std
    high_cols = [c for c in df.columns if 'high' in c.lower()]
    low_cols = [c for c in df.columns if 'low' in c.lower()]
    if high_cols and low_cols:
        high_col = high_cols[0]
        low_col = low_cols[0]
        high_low = df[high_col] - df[low_col]
        high_close = (df[high_col] - prices.shift()).abs()
        low_close = (df[low_col] - prices.shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(window=14).mean()
    else:
        # Fallback: use standard deviation as volatility proxy
        df['ATR'] = prices.rolling(window=14).std()
    
    # Calculate returns for performance analysis
    df['Returns'] = prices.pct_change()
    df['Cumulative_Returns'] = (1 + df['Returns']).cumprod() - 1
    
    return df


def wrangle_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and transform the data.
    Removes bad data points and handles missing values.
    """
    # Drop rows that are completely empty
    df = df.dropna(how='all')
    
    # Handle missing values differently for different column types
    price_cols = [c for c in df.columns if 'Close' in c]
    # Indicators might not exist yet, so check carefully
    indicator_cols = [c for c in df.columns if c in ['RSI', 'MACD', 'MA_20', 'MA_50']]
    
    # For prices, forward fill makes sense (use last known price)
    if price_cols:
        df[price_cols] = df[price_cols].ffill()
    
    # For indicators, backward fill then zero if still missing
    if indicator_cols:
        df[indicator_cols] = df[indicator_cols].bfill().fillna(0)
    
    # Remove extreme outliers - prices more than 3 std devs from mean
    # This helps with data quality issues
    for col in price_cols:
        if len(df) > 0:
            mean = df[col].mean()
            std = df[col].std()
            if std > 0:  # Avoid division issues
                df = df[(df[col] >= mean - 3*std) & (df[col] <= mean + 3*std)]
    
    # Add some derived features that might be useful
    if 'BTCUSDT_Close' in df.columns:
        df['Price_Change'] = df['BTCUSDT_Close'].diff()
        df['Price_Change_Pct'] = df['BTCUSDT_Close'].pct_change() * 100
    
    return df


def analyze_market_trends(df: pd.DataFrame, price_col: str = "BTCUSDT_Close") -> Dict:
    """
    My analysis function - extracts key statistics and signals from the data.
    Returns a dictionary with price stats, returns, and indicator signals.
    """
    prices = df[price_col].dropna()
    
    # Basic price statistics
    analysis = {
        'price_stats': {
            'mean': float(prices.mean()),
            'std': float(prices.std()),
            'min': float(prices.min()),
            'max': float(prices.max()),
            'current': float(prices.iloc[-1]),
        },
        'returns_stats': {},
        'indicators': {}
    }
    
    # Calculate return statistics if available
    if 'Returns' in df.columns:
        returns = df['Returns'].dropna()
        if len(returns) > 0:
            analysis['returns_stats'] = {
                'mean_daily_return': float(returns.mean() * 100),
                'volatility': float(returns.std() * 100),
                'total_return': float(df['Cumulative_Returns'].iloc[-1] * 100) if 'Cumulative_Returns' in df.columns else None,
            }
    
    # RSI analysis - check for overbought/oversold conditions
    if 'RSI' in df.columns:
        rsi = df['RSI'].dropna()
        if len(rsi) > 0:
            current_rsi = rsi.iloc[-1]
            analysis['indicators']['RSI'] = {
                'current': float(current_rsi),
                'mean': float(rsi.mean()),
                'oversold_signal': current_rsi < 30,  # RSI < 30 suggests oversold
                'overbought_signal': current_rsi > 70,  # RSI > 70 suggests overbought
            }
    
    # MACD analysis - look for bullish/bearish crossovers
    if 'MACD' in df.columns and 'MACD_Signal' in df.columns:
        if len(df) > 0:
            macd = df['MACD'].iloc[-1]
            signal = df['MACD_Signal'].iloc[-1]
            analysis['indicators']['MACD'] = {
                'current': float(macd) if pd.notna(macd) else None,
                'signal': float(signal) if pd.notna(signal) else None,
                'bullish_cross': macd > signal if (pd.notna(macd) and pd.notna(signal)) else False,
            }
    
    return analysis


def visualize_data(df: pd.DataFrame, price_col: str = "BTCUSDT_Close", save_path: str = "analysis_plot.png"):
    """
    Create visualization plots showing price, RSI, and MACD indicators.
    Saves the plot to a file.
    """
    # Create 3 subplots stacked vertically
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle('Financial Data Analysis Dashboard', fontsize=16, fontweight='bold')
    
    # Top plot: Price with moving averages
    ax1 = axes[0]
    ax1.plot(df.index, df[price_col], label='Price', linewidth=2, color='#2E86AB')
    
    # Add moving averages if they exist
    if 'MA_20' in df.columns:
        ax1.plot(df.index, df['MA_20'], label='MA 20', alpha=0.7, color='orange', linestyle='--')
    if 'MA_50' in df.columns:
        ax1.plot(df.index, df['MA_50'], label='MA 50', alpha=0.7, color='red', linestyle='--')
    
    # Add Bollinger Bands as shaded area
    if 'BB_Upper' in df.columns and 'BB_Lower' in df.columns:
        ax1.fill_between(df.index, df['BB_Upper'], df['BB_Lower'], 
                        alpha=0.2, color='gray', label='Bollinger Bands')
    
    ax1.set_title('Price Chart with Technical Indicators', fontweight='bold')
    ax1.set_ylabel('Price (USDT)', fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Middle plot: RSI indicator
    ax2 = axes[1]
    if 'RSI' in df.columns:
        ax2.plot(df.index, df['RSI'], label='RSI', color='purple', linewidth=2)
        # Add reference lines
        ax2.axhline(y=70, color='r', linestyle='--', alpha=0.5, label='Overbought (70)')
        ax2.axhline(y=30, color='g', linestyle='--', alpha=0.5, label='Oversold (30)')
        ax2.fill_between(df.index, 30, 70, alpha=0.1, color='gray')
    
    ax2.set_title('RSI Indicator', fontweight='bold')
    ax2.set_ylabel('RSI', fontweight='bold')
    ax2.set_ylim(0, 100)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # Bottom plot: MACD
    ax3 = axes[2]
    if 'MACD' in df.columns and 'MACD_Signal' in df.columns:
        ax3.plot(df.index, df['MACD'], label='MACD', color='blue', linewidth=2)
        ax3.plot(df.index, df['MACD_Signal'], label='Signal', color='red', linewidth=2)
        # Add histogram bars
        if 'MACD_Hist' in df.columns:
            colors = ['green' if x >= 0 else 'red' for x in df['MACD_Hist']]
            ax3.bar(df.index, df['MACD_Hist'], alpha=0.3, color=colors, label='Histogram')
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    ax3.set_title('MACD Indicator', fontweight='bold')
    ax3.set_ylabel('MACD', fontweight='bold')
    ax3.set_xlabel('Date', fontweight='bold')
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3)
    
    # Save and display
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Visualization saved to {save_path}")
    plt.close()  # Close to avoid displaying in non-interactive mode


# Main execution
if __name__ == "__main__":
    print("=" * 60)
    print("Financial Data Analysis Script")
    print("=" * 60)
    
    # Step 1: Create and ingest data from CSV
    # For this demo, I'll generate sample data and save to CSV
    # In real use, you'd load from an existing CSV file
    print("\n[Step 1] Creating sample CSV data...")
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    np.random.seed(42)  # For reproducibility
    base_price = 50000
    # Simulate realistic price movements with 2% daily volatility
    returns = np.random.randn(len(dates)) * 0.02
    prices = base_price * (1 + returns).cumprod()
    
    # Create DataFrame with OHLC-like data
    sample_df = pd.DataFrame({
        'Date': dates,
        'BTCUSDT_Close': prices,
        'BTCUSDT_High': prices * (1 + np.abs(np.random.randn(len(dates)) * 0.01)),
        'BTCUSDT_Low': prices * (1 - np.abs(np.random.randn(len(dates)) * 0.01)),
        'BTCUSDT_Volume': np.random.uniform(1000, 5000, len(dates))
    })
    sample_df = sample_df.set_index('Date')
    csv_path = 'sample_financial_data.csv'
    sample_df.to_csv(csv_path)
    print(f"  Created sample data: {csv_path}")
    
    # Load the CSV file
    df = ingest_data_from_csv(csv_path)
    
    # Step 2: Clean and wrangle the data
    print("\n[Step 2] Wrangling data...")
    df = wrangle_data(df)
    print(f"  ✓ Data cleaned: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Step 3: Calculate technical indicators
    print("\n[Step 3] Calculating technical indicators...")
    df = calculate_technical_indicators(df)
    print("  ✓ Indicators calculated: RSI, MACD, Moving Averages, Bollinger Bands, ATR")
    
    # Step 4: Analyze the data
    print("\n[Step 4] Analyzing market trends...")
    analysis = analyze_market_trends(df)
    print(f"  Current Price: ${analysis['price_stats']['current']:,.2f}")
    print(f"  Price Range: ${analysis['price_stats']['min']:,.2f} - ${analysis['price_stats']['max']:,.2f}")
    
    if analysis['returns_stats']:
        print(f"  Mean Daily Return: {analysis['returns_stats']['mean_daily_return']:.2f}%")
        print(f"  Volatility: {analysis['returns_stats']['volatility']:.2f}%")
        if analysis['returns_stats']['total_return'] is not None:
            print(f"  Total Return: {analysis['returns_stats']['total_return']:.2f}%")
    
    if 'RSI' in analysis['indicators']:
        rsi_info = analysis['indicators']['RSI']
        print(f"\n  RSI Analysis:")
        print(f"    Current RSI: {rsi_info['current']:.2f}")
        print(f"    Oversold Signal: {rsi_info['oversold_signal']}")
        print(f"    Overbought Signal: {rsi_info['overbought_signal']}")
    
    if 'MACD' in analysis['indicators']:
        macd_info = analysis['indicators']['MACD']
        if macd_info['current'] is not None:
            print(f"\n  MACD Analysis:")
            print(f"    MACD: {macd_info['current']:.4f}")
            print(f"    Signal: {macd_info['signal']:.4f}")
            print(f"    Bullish Cross: {macd_info['bullish_cross']}")
    
    # Step 5: Create visualizations
    print("\n[Step 5] Creating visualizations...")
    visualize_data(df, save_path="financial_analysis.png")
    
    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)
    print(f"Final dataset: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"Generated files: {csv_path}, financial_analysis.png")
