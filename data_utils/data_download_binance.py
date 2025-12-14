#!/usr/bin/env python3
"""
从 Binance API 获取 BTCUSDT 的所有历史日K线数据
"""

import requests
import pandas as pd
import time
from datetime import datetime

def get_all_klines(symbol="BTCUSDT", interval="1d", start_time=None):
    """
    获取指定交易对的所有历史K线数据
    
    Args:
        symbol: 交易对，如 BTCUSDT
        interval: K线间隔，1d=日线, 1h=小时线等
        start_time: 开始时间戳(毫秒)，默认从最早数据开始
    
    Returns:
        DataFrame: 包含所有历史数据的DataFrame
    """
    # url = "https://api.binance.us/api/v3/klines"
    url = "https://api.binance.com/api/v3/klines"
    all_data = []
    
    # BTCUSDT 上线时间大约是 2017年8月17日
    if start_time is None:
        start_time = int(datetime(2017, 8, 17).timestamp() * 1000)
    
    end_time = int(datetime.now().timestamp() * 1000)
    
    print(f"开始获取 {symbol} 的历史数据...")
    print(f"时间范围: {datetime.fromtimestamp(start_time/1000)} 至今")
    print("-" * 50)
    
    request_count = 0
    
    while start_time < end_time:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_time,
            "limit": 1000  # 每次最多1000条
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if not data:
                break
            
            all_data.extend(data)
            request_count += 1
            
            # 更新起始时间为最后一条数据的收盘时间+1
            start_time = data[-1][6] + 1
            
            # 显示进度
            last_date = datetime.fromtimestamp(data[-1][0] / 1000).strftime('%Y-%m-%d')
            print(f"已获取 {len(all_data)} 条数据，最新日期: {last_date}")
            
            # 控制请求频率，避免被限制
            time.sleep(0.2)
            
        except requests.exceptions.RequestException as e:
            print(f"请求错误: {e}")
            print("等待5秒后重试...")
            time.sleep(5)
            continue
    
    print("-" * 50)
    print(f"数据获取完成！共发送 {request_count} 次请求")
    
    # 转换为DataFrame
    df = pd.DataFrame(all_data, columns=[
        'open_time',      # 开盘时间
        'open',           # 开盘价
        'high',           # 最高价
        'low',            # 最低价
        'close',          # 收盘价
        'volume',         # 成交量
        'close_time',     # 收盘时间
        'quote_volume',   # 成交额
        'trades',         # 成交笔数
        'taker_buy_base', # 主动买入成交量
        'taker_buy_quote',# 主动买入成交额
        'ignore'          # 忽略
    ])
    
    # 数据类型转换
    numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_volume', 
                    'taker_buy_base', 'taker_buy_quote']
    df[numeric_cols] = df[numeric_cols].astype(float)
    df['trades'] = df['trades'].astype(int)
    
    # 转换时间格式
    df['date'] = pd.to_datetime(df['open_time'], unit='ms')
    df['close_date'] = pd.to_datetime(df['close_time'], unit='ms')
    
    # 重新排列列顺序
    df = df[['date', 'open', 'high', 'low', 'close', 'volume', 
             'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote',
             'open_time', 'close_time']]
    
    return df


def main():
    # 获取所有历史数据
    df = get_all_klines(symbol="BTCUSDT", interval="1d")
    
    # 显示数据信息
    print(f"\n数据概览:")
    print(f"  总天数: {len(df)}")
    print(f"  起始日期: {df['date'].min().strftime('%Y-%m-%d')}")
    print(f"  结束日期: {df['date'].max().strftime('%Y-%m-%d')}")
    print(f"  历史最高价: ${df['high'].max():,.2f}")
    print(f"  历史最低价: ${df['low'].min():,.2f}")
    
    # 保存为CSV文件
    output_file = "btc_daily_history.csv"
    df.to_csv(output_file, index=False)
    print(f"\n数据已保存至: {output_file}")
    
    # 显示前几行和后几行
    print("\n最早5天数据:")
    print(df.head().to_string(index=False))
    print("\n最近5天数据:")
    print(df.tail().to_string(index=False))
    
    return df


if __name__ == "__main__":
    df = main()