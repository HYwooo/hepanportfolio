# generate_report.py

import os
import requests
import pandas as pd
import numpy as np
import quantstats as qs
import json
import random
import sys
from datetime import datetime, timedelta, timezone
from playwright.sync_api import sync_playwright
import threading
import http.server
import socketserver

# --- 配置参数 ---

API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")
if not API_KEY:
    raise ValueError("ALPHAVANTAGE_API_KEY environment variable not set.")
CACHE_DIR = "data_cache"
TICKERS = ['513110.SHH', '518660.SHH', '159649.SHZ', '515450.SHH']
BENCHMARK_TICKER = '000300.SHH'
WEIGHTS = [0.25, 0.25, 0.25, 0.25]
INITIAL_CAPITAL = 10000
START_DATE = "2025-09-22"
RISK_FREE_RATE = 0.02
OUTPUT_PNG_PATH = "pages/portfolio_chart.png"
OUTPUT_HTML_PATH = "pages/index.html"
OUTPUT_JSON_PATH = "pages/data.json"


# --- 数据获取模块 (保持不变) ---
def fetch_data_from_api(ticker, output_size='full'):
    print(f"\n--- Attempting to fetch data for {ticker} from API (outputsize={output_size}) ---")
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&apikey={API_KEY}'
    if output_size == 'full':
        url += '&outputsize=full'
    print(f"Requesting URL: {url}")
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        data = r.json()
        if "Time Series (Daily)" not in data or not data["Time Series (Daily)"]:
            print(f"ERROR: 'Time Series (Daily)' not found in response for {ticker}.")
            print("Full API Response:", json.dumps(data, indent=2))
            return None
        df = pd.DataFrame.from_dict(data['Time Series (Daily)'], orient='index')
        df = df.astype(float)
        df.index = pd.to_datetime(df.index)
        print(f"Successfully fetched {len(df)} data points for {ticker}.")
        return df['4. close'].sort_index()
    except Exception as e:
        print(f"Request failed for {ticker}: {e}")
        return None

def get_data(ticker):
    # ... (代码不变)
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    cache_path = os.path.join(CACHE_DIR, f"{ticker.replace('.', '_')}.csv")
    if os.path.exists(cache_path):
        try:
            cached_df = pd.read_csv(cache_path, index_col='date', parse_dates=True)
            if cached_df.empty:
                 raise ValueError("Cache file is empty.")
            last_cached_date = cached_df.index.max().date()
            today_utc8 = (datetime.now(timezone.utc) + timedelta(hours=8)).date()
            if last_cached_date >= today_utc8:
                print(f"Cache for {ticker} is already up-to-date for today ({last_cached_date}). Skipping API call.")
                return cached_df['close']
            print(f"Cache for {ticker} is not current. Attempting incremental update.")
            api_data_update = fetch_data_from_api(ticker, output_size='compact')
            if api_data_update is not None and not api_data_update.empty:
                update_df = pd.DataFrame({'close': api_data_update})
                update_df.index.name = 'date'
                combined_df = pd.concat([cached_df, update_df])
                combined_df = combined_df[~combined_df.index.duplicated(keep='first')]
                combined_df.sort_index(inplace=True)
                combined_df.to_csv(cache_path)
                print(f"Cache for {ticker} successfully updated.")
                return combined_df['close']
            else:
                print(f"API update failed for {ticker}. Using stale cache as fallback.")
                return cached_df['close']
        except Exception as e:
            print(f"Could not read or process cache file for {ticker}: {e}. Falling back to full fetch.")
            pass
    print(f"Cache not found for {ticker} or update failed. Performing full fetch.")
    api_data_full = fetch_data_from_api(ticker, output_size='full')
    if api_data_full is not None and not api_data_full.empty:
        df_to_save = pd.DataFrame({'close': api_data_full})
        df_to_save.index.name = 'date'
        df_to_save.to_csv(cache_path)
        print(f"Saved new full data for {ticker} to cache.")
        return api_data_full
    print(f"CRITICAL: Failed to get any data for {ticker}.")
    return None

# --- 回测模块 (保持不变) ---
def run_backtest(assets_data, benchmark_data):
    print("Running backtest..."); portfolio_data = pd.concat(assets_data, axis=1); portfolio_data.columns = TICKERS
    portfolio_data = portfolio_data.loc[START_DATE:]; portfolio_data = portfolio_data.ffill().bfill()
    if portfolio_data.empty: return None, None
    returns = portfolio_data.pct_change(); portfolio_value = pd.Series(index=portfolio_data.index, dtype=float); portfolio_value.iloc[0] = INITIAL_CAPITAL
    asset_values = portfolio_value.iloc[0] * np.array(WEIGHTS); last_rebalance_year = portfolio_value.index[0].year
    for i in range(1, len(portfolio_data)):
        asset_values *= (1 + returns.iloc[i]); current_date = portfolio_data.index[i]
        if current_date.year != last_rebalance_year:
            print(f"Rebalancing for year {current_date.year}..."); portfolio_total_value = np.sum(asset_values); asset_values = portfolio_total_value * np.array(WEIGHTS)
            last_rebalance_year = current_date.year
        portfolio_value.iloc[i] = np.sum(asset_values)
    portfolio_returns = portfolio_value.pct_change().dropna()
    full_benchmark_returns = benchmark_data.pct_change()
    benchmark_returns = full_benchmark_returns.loc[portfolio_returns.index].dropna()
    common_index = portfolio_returns.index.intersection(benchmark_returns.index)
    portfolio_returns = portfolio_returns.loc[common_index]
    benchmark_returns = benchmark_returns.loc[common_index]
    return portfolio_returns, benchmark_returns

# --- JSON 数据生成模块 (保持不变) ---
def generate_data_json(portfolio_returns=None, benchmark_returns=None, is_future=False):
    print("Generating data.json...")
    output_data = {}
    if is_future or portfolio_returns is None or portfolio_returns.empty:
        output_data = {
            "status": "future",
            "startDate": START_DATE,
            "lastChecked": datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S') + " UTC"
        }
    else:
        metrics = {
            'Total Return': qs.stats.comp(portfolio_returns),
            'Annualized Return': qs.stats.cagr(portfolio_returns),
            'Annualized Volatility': qs.stats.volatility(portfolio_returns, periods=252),
            'Max Drawdown': qs.stats.max_drawdown(portfolio_returns),
            'Sharpe Ratio': qs.stats.sharpe(portfolio_returns, rf=RISK_FREE_RATE),
            'Sortino Ratio': qs.stats.sortino(portfolio_returns, rf=RISK_FREE_RATE),
            'Benchmark Total Return': qs.stats.comp(benchmark_returns),
            'Benchmark Annualized Return': qs.stats.cagr(benchmark_returns),
            'Benchmark Annualized Volatility': qs.stats.volatility(benchmark_returns, periods=252),
            'Benchmark Max Drawdown': qs.stats.max_drawdown(benchmark_returns),
            'Benchmark Sharpe Ratio': qs.stats.sharpe(benchmark_returns, rf=RISK_FREE_RATE),
            'Benchmark Sortino Ratio': qs.stats.sortino(benchmark_returns, rf=RISK_FREE_RATE)
        }
        
        portfolio_cumulative = (1 + portfolio_returns).cumprod() * INITIAL_CAPITAL
        benchmark_cumulative = (1 + benchmark_returns).cumprod() * INITIAL_CAPITAL

        chart_data_portfolio = [{"time": date.strftime('%Y-%m-%d'), "value": round(value, 2)} for date, value in portfolio_cumulative.items()]
        chart_data_benchmark = [{"time": date.strftime('%Y-%m-%d'), "value": round(value, 2)} for date, value in benchmark_cumulative.items()]
        
        output_data = {
            "status": "success",
            "lastUpdated": datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S') + " UTC",
            "initialCapital": INITIAL_CAPITAL,
            "benchmarkTicker": BENCHMARK_TICKER.split('.')[0],
            "metrics": metrics,
            "chartData": {
                "portfolio": chart_data_portfolio,
                "benchmark": chart_data_benchmark
            }
        }
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)
    print(f"Successfully generated {OUTPUT_JSON_PATH}")

# --- PNG 生成函数 (完全替换为这个新版本) ---
def generate_png_from_html(html_path=OUTPUT_HTML_PATH, png_path=OUTPUT_PNG_PATH):
    """通过启动本地服务器并使用Playwright访问来对图表进行截图"""
    print(f"Starting PNG generation from {html_path}...")
    os.makedirs(os.path.dirname(png_path), exist_ok=True)
    
    PORT = 8008 # Use an uncommon port number
    # SimpleHTTPRequestHandler 会自动寻找当前目录下的文件
    Handler = http.server.SimpleHTTPRequestHandler
    
    # 我们需要在项目根目录运行服务器，所以暂时切换目录
    current_dir = os.getcwd()
    # 假设你的 pages 目录在项目根目录下
    # os.chdir(os.path.dirname(html_path))
    # 更新：更好的方法是不切换目录，直接从根目录访问
    
    httpd = socketserver.TCPServer(("", PORT), Handler)
    
    # 启动一个后台线程来运行服务器
    server_thread = threading.Thread(target=httpd.serve_forever)
    server_thread.daemon = True  # 确保主线程退出时，这个线程也会退出
    server_thread.start()
    print(f"Local server started at http://localhost:{PORT}")
    
    success = False
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(
                viewport={"width": 1024, "height": 768},
                device_scale_factor=2 # 提高截图清晰度
            )
            # 访问由本地服务器提供的页面
            # html_path 已经是 'pages/index.html'
            page_url = f'http://localhost:{PORT}/{html_path}'
            print(f"Playwright going to: {page_url}")
            page.goto(page_url, wait_until='networkidle')
            
            # 等待图表容器元素出现
            chart_element = page.locator('#chart-container')
            chart_element.wait_for(state='visible', timeout=10000) # 等待图表完全加载
            
            page.wait_for_timeout(3000) # 额外等待5秒，确保图表完全加载 
            print("Taking screenshot...")
            chart_element.screenshot(path=png_path)
            browser.close()
            print(f"Successfully generated PNG: {png_path}")
            success = True
    except Exception as e:
        print(f"Error during PNG generation: {e}")
        success = False
    finally:
        # 无论成功与否，都关闭服务器
        print("Shutting down local server...")
        httpd.shutdown()
        httpd.server_close()
        # os.chdir(current_dir) # 切换回原来的目录
        print("Server stopped.")
        
    return success

# --- 主执行逻辑 (保持不变) ---
if __name__ == "__main__":
    current_utc = datetime.now(timezone.utc)
    utc_plus_8_time = current_utc + timedelta(hours=8)
    if not (19 <= utc_plus_8_time.hour < 23):
        print(f"Execution stopped. Current time {utc_plus_8_time.strftime('%Y-%m-%d %H:%M:%S')} UTC+8 is outside the allowed window (19:00 - 23:00).")
        sys.exit()
    print(f"Current time {utc_plus_8_time.strftime('%Y-%m-%d %H:%M:%S')} UTC+8 is within the allowed window. Starting process...")
    if not API_KEY or API_KEY == "YOUR_API_KEY_HERE": 
        raise ValueError("Alpha Vantage API Key not found. Please set it as an environment variable.")
    all_tickers = TICKERS + [BENCHMARK_TICKER]
    all_data = {ticker: get_data(ticker) for ticker in all_tickers}
    if any(data is None for data in all_data.values()): 
        print("\nCritical Error: Failed to get data for one or more tickers.")
    else:
        assets_data = [all_data[ticker] for ticker in TICKERS]
        benchmark_data = all_data[BENCHMARK_TICKER]
        portfolio_returns, benchmark_returns = None, None
        try:
            portfolio_returns, benchmark_returns = run_backtest(assets_data, benchmark_data)
            print("Backtest completed successfully.")
        except Exception as e:
            print(f"Error during backtest: {e}")
        if portfolio_returns is None or portfolio_returns.empty:
            print("Backtest resulted in no data, likely because start date is in the future.")
            generate_data_json(is_future=True)
        else:
            generate_data_json(portfolio_returns, benchmark_returns)
            if os.path.exists(OUTPUT_HTML_PATH):
                generate_png_from_html()
            else:
                print(f"Warning: {OUTPUT_HTML_PATH} not found. Skipping PNG generation.")