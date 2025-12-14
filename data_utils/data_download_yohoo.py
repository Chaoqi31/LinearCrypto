#!/usr/bin/env python3
"""
从 Yahoo Finance 获取比特币完整历史数据
数据追溯到 2014年9月，美国可用
"""

import yfinance as yf
import pandas as pd
from datetime import datetime


def get_btc_history(start_date="2014-01-01", end_date=None):
    """
    从 Yahoo Finance 获取 BTC-USD 历史数据
    
    Args:
        start_date: 开始日期，默认2014-01-01
        end_date: 结束日期，默认今天
    
    Returns:
        DataFrame: 历史价格数据
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    
    print(f"从 Yahoo Finance 获取 BTC-USD 数据...")
    print(f"时间范围: {start_date} 至 {end_date}")
    print("-" * 50)
    
    # 下载数据
    df = yf.download("BTC-USD", start=start_date, end=end_date, progress=True)
    
    # 重置索引，将日期变为列
    df = df.reset_index()
    
    # 处理多级列索引（新版yfinance可能返回MultiIndex）
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if col[1] == '' else col[0] for col in df.columns]
    
    # 统一列名为小写
    df.columns = [col.lower().replace(' ', '_') for col in df.columns]
    
    # 确保有必需的列
    required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
    for col in required_cols:
        if col not in df.columns:
            print(f"警告: 缺少列 {col}")
    
    # 只保留需要的列
    available_cols = [col for col in required_cols if col in df.columns]
    df = df[available_cols]
    
    return df


def main():
    # 获取所有历史数据
    df = get_btc_history()
    
    # 显示数据信息
    print(f"\n数据概览:")
    print(f"  总天数: {len(df)}")
    print(f"  起始日期: {df['date'].min().strftime('%Y-%m-%d')}")
    print(f"  结束日期: {df['date'].max().strftime('%Y-%m-%d')}")
    print(f"  历史最高价: ${df['high'].max():,.2f}")
    print(f"  历史最低价: ${df['low'].min():,.2f}")
    
    # 保存为CSV
    output_file = "btc_daily_history_yahoo.csv"
    df.to_csv(output_file, index=False)
    print(f"\n数据已保存至: {output_file}")
    
    # 显示数据样例
    print("\n最早5天数据:")
    print(df.head().to_string(index=False))
    print("\n最近5天数据:")
    print(df.tail().to_string(index=False))
    
    return df


if __name__ == "__main__":
    df = main()