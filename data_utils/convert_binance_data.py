"""
Convert Binance daily BTC data to CryptoMamba format.

Input format (btc_daily_history_binance.csv):
    Date,Open,High,Low,Close,Volume,Quote_Volume,Trades,Taker_Buy_Base,Taker_Buy_Quote,Open_Time,Close_Time

Output format (binance_processed.csv):
    ,Open,High,Low,Close,Volume,Timestamp
"""

import os
import pandas as pd
from pathlib import Path


def convert_binance_data(input_path: str, output_path: str) -> None:
    """
    Convert Binance CSV data to CryptoMamba format.

    Args:
        input_path: Path to btc_daily_history_binance.csv
        output_path: Path to output binance_processed.csv
    """
    # Read Binance data
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} rows from {input_path}")

    # Select and rename columns
    # Open_Time is in milliseconds, convert to seconds
    processed_df = pd.DataFrame({
        'Open': df['Open'],
        'High': df['High'],
        'Low': df['Low'],
        'Close': df['Close'],
        'Volume': df['Volume'],
        'Timestamp': df['Open_Time'] // 1000  # ms -> seconds
    })

    # Sort by timestamp ascending (oldest first)
    processed_df = processed_df.sort_values('Timestamp').reset_index(drop=True)

    # Save to CSV with index (matches train.csv format)
    processed_df.to_csv(output_path)
    print(f"Saved {len(processed_df)} rows to {output_path}")

    # Print sample
    print("\nFirst 5 rows:")
    print(processed_df.head())
    print("\nLast 5 rows:")
    print(processed_df.tail())

    # Print date range
    from datetime import datetime
    start_date = datetime.fromtimestamp(processed_df['Timestamp'].iloc[0])
    end_date = datetime.fromtimestamp(processed_df['Timestamp'].iloc[-1])
    print(f"\nDate range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")


if __name__ == "__main__":
    # Get project root
    root = Path(__file__).parent.parent

    input_file = root / "data" / "btc_daily_history_binance.csv"
    output_file = root / "data" / "binance_processed.csv"

    convert_binance_data(str(input_file), str(output_file))
