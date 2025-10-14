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

# --- 配置参数 ---
API_KEY ="TQONN184ZFV8GHJG"
#API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY') if os.getenv('ALPHA_VANTAGE_API_KEY') else str(random.randint(114514, 1919810114514))
CACHE_DIR = "data_cache"
TICKERS = ['513110.SHH', '518660.SHH', '159649.SHZ', '515450.SHH']
BENCHMARK_TICKER = '000300.SHH'
WEIGHTS = [0.25, 0.25, 0.25, 0.25]
INITIAL_CAPITAL = 10000
START_DATE = "2025-09-22"
RISK_FREE_RATE = 0.02
OUTPUT_PNG_PATH = "./pages/portfolio_chart.png"
OUTPUT_HTML_PATH = "./pages/index.html" # This is used by the PNG generator
OUTPUT_JSON_PATH = "./pages/data.json" # New path for dynamic data

# --- 数据获取模块 ---
def fetch_data_from_api(ticker, output_size='full'):
    """
    从Alpha Vantage API获取数据。
    :param ticker: 股票代码
    :param output_size: 'full' 获取全部历史数据，'compact' (默认) 获取最近100个数据点。
    """
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
    """
    获取单个 Ticker 的数据，优先使用缓存，并实现增量更新逻辑。
    """
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


# --- 回测模块 ---
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

# --- JSON 数据生成模块 (新) ---
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
            "benchmarkTicker": BENCHMARK_TICKER.split('.')[0], # e.g., "000300"
            "metrics": metrics,
            "chartData": {
                "portfolio": chart_data_portfolio,
                "benchmark": chart_data_benchmark
            }
        }
    
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)
    print(f"Successfully generated {OUTPUT_JSON_PATH}")

# --- PNG 生成函数 ---
def generate_png_from_html(html_path=OUTPUT_HTML_PATH, png_path=OUTPUT_PNG_PATH):
    """使用Playwright对本地HTML文件中的图表进行截图"""
    print(f"Starting PNG generation from {html_path}...")
    # Make sure the target directory exists
    os.makedirs(os.path.dirname(png_path), exist_ok=True)
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            context = browser.new_context(
                viewport={"width": 1920, "height": 1080},
                device_scale_factor=3
            )
            page = browser.new_page()
            absolute_html_path = os.path.abspath(html_path)
            page.goto(f'file://{absolute_html_path}')
            page.wait_for_load_state('networkidle') 
            page.wait_for_timeout(3000) # Give more time for JS to fetch and render
            chart_element = page.locator('#chart-container')
            chart_element.screenshot(path=png_path)
            browser.close()
            print(f"Successfully generated PNG: {png_path}")
            return True
    except Exception as e:
        print(f"Error during PNG generation: {e}")
        return False
    
# --- 主执行逻辑 ---
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
            # PNG generation still needs an HTML file to render from
            if os.path.exists(OUTPUT_HTML_PATH):
                generate_png_from_html()
            else:
                print(f"Warning: {OUTPUT_HTML_PATH} not found. Skipping PNG generation.")