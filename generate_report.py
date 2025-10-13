import os
import requests
import pandas as pd
import numpy as np
import quantstats as qs
import json
from datetime import datetime
from playwright.sync_api import sync_playwright

# --- 配置参数 ---
API_KEY = "114514"#os.getenv('ALPHA_VANTAGE_API_KEY') 
CACHE_DIR = "data_cache"
TICKERS = ['513110.SHH', '518660.SHH', '159649.SHZ', '515450.SHH']
BENCHMARK_TICKER = '000300.SHH'
WEIGHTS = [0.25, 0.25, 0.25, 0.25]
INITIAL_CAPITAL = 10000
START_DATE = "2025-09-23" 
RISK_FREE_RATE = 0.02
OUTPUT_PNG_PATH = "portfolio_chart.png" # 定义输出PNG的文件名

# --- 数据获取模块 ---
def fetch_data_from_api(ticker):
    print(f"\n--- Attempting to fetch data for {ticker} from API ---")
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&outputsize=full&apikey={API_KEY}'
    print(f"Requesting URL: {url}")
    try:
        r = requests.get(url, timeout=30); r.raise_for_status(); data = r.json()
        if "Time Series (Daily)" not in data or not data["Time Series (Daily)"]:
            print(f"ERROR: 'Time Series (Daily)' not found in response for {ticker}."); print("Full API Response:", json.dumps(data, indent=2)); return None
        df = pd.DataFrame.from_dict(data['Time Series (Daily)'], orient='index'); df = df.astype(float); df.index = pd.to_datetime(df.index)
        print(f"Successfully fetched {len(df)} data points for {ticker}.")
        return df['4. close'].sort_index()
    except Exception as e:
        print(f"Request failed for {ticker}: {e}"); return None

def get_data(ticker):
    if not os.path.exists(CACHE_DIR): os.makedirs(CACHE_DIR)
    cache_path = os.path.join(CACHE_DIR, f"{ticker.replace('.', '_')}.csv")
    if os.path.exists(cache_path):
        try:
            cached_df = pd.read_csv(cache_path, index_col='date', parse_dates=True)
            if not cached_df.empty and (datetime.now().date() == cached_df.index.max().date()):
                print(f"Using up-to-date cache for {ticker}."); return cached_df['close']
        except Exception as e:
            print(f"Could not read cache file for {ticker}: {e}")
    api_data = fetch_data_from_api(ticker)
    if api_data is not None and not api_data.empty:
        df_to_save = pd.DataFrame({'close': api_data}); df_to_save.index.name = 'date'; df_to_save.to_csv(cache_path)
        print(f"Saved new data for {ticker} to cache."); return api_data
    if os.path.exists(cache_path):
        print(f"API fetch failed for {ticker}. Using stale cache as fallback."); return pd.read_csv(cache_path, index_col='date', parse_dates=True)['close']
    return None

# --- 回测模块 (已修正日期对齐) ---
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

# --- HTML 报告生成模块 (已最终修正 JS 调用 和 指标) ---
def generate_html_report(portfolio_returns=None, benchmark_returns=None, is_future=False):
    print("Generating Web3-style HTML report with custom legend...")
    if is_future or portfolio_returns is None or portfolio_returns.empty:
        html_content = f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><title>Portfolio Report</title><script src="https://cdn.tailwindcss.com"></script></head><body class="bg-slate-900 text-slate-300 flex items-center justify-center min-h-screen"><div class="text-center p-8 bg-slate-800/50 rounded-lg backdrop-blur-sm ring-1 ring-white/10"><h1 class="text-3xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-purple-500">报告将在未来生成</h1><p class="mt-4">投资组合的起始日期设置为 {START_DATE}。</p><p class="mt-2">请在该日期之后查看此页面以获取详细的性能报告。</p><p class="text-xs text-slate-500 mt-6">最后检查时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC</p></div></body></html>"""
    else:
        metrics = {
            'Total Return': qs.stats.comp(portfolio_returns), 'Annualized Return': qs.stats.cagr(portfolio_returns),
            'Annualized Volatility': qs.stats.volatility(portfolio_returns, periods=252), 'Max Drawdown': qs.stats.max_drawdown(portfolio_returns),
            'Sharpe Ratio': qs.stats.sharpe(portfolio_returns, rf=RISK_FREE_RATE), 'Sortino Ratio': qs.stats.sortino(portfolio_returns, rf=RISK_FREE_RATE),
            'Benchmark Total Return': qs.stats.comp(benchmark_returns), 'Benchmark Annualized Return': qs.stats.cagr(benchmark_returns),
            'Benchmark Annualized Volatility': qs.stats.volatility(benchmark_returns, periods=252), 'Benchmark Max Drawdown': qs.stats.max_drawdown(benchmark_returns),
            'Benchmark Sharpe Ratio': qs.stats.sharpe(benchmark_returns, rf=RISK_FREE_RATE), 'Benchmark Sortino Ratio': qs.stats.sortino(benchmark_returns, rf=RISK_FREE_RATE)
        }
        portfolio_cumulative = (1 + portfolio_returns).cumprod() * INITIAL_CAPITAL
        benchmark_cumulative = (1 + benchmark_returns).cumprod() * INITIAL_CAPITAL
        chart_data_portfolio = [{"time": str(date.date()), "value": round(value, 2)} for date, value in portfolio_cumulative.items()]
        chart_data_benchmark = [{"time": str(date.date()), "value": round(value, 2)} for date, value in benchmark_cumulative.items()]
        html_content = f"""
        <!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Portfolio Performance Dashboard</title>
            <script src="https://cdn.tailwindcss.com"></script><script src="https://unpkg.com/lightweight-charts/dist/lightweight-charts.standalone.production.js"></script>
            <style>
                body::before {{ content: ''; position: fixed; left: 0; top: 0; width: 100%; height: 100%; will-change: transform; background: radial-gradient(circle at 20% 20%, rgba(10, 178, 240, 0.2), transparent 30%), radial-gradient(circle at 80% 70%, rgba(168, 85, 247, 0.2), transparent 30%); filter: blur(100px); z-index: -1; }}
                #chart-container {{ position: relative; }}
            </style>
        </head>
        <body class="bg-slate-900 text-slate-200 font-sans">
            <main class="min-h-screen flex items-center justify-center p-4">
                <div class="w-full max-w-4xl rounded-2xl bg-slate-800/50 p-6 md:p-8 shadow-2xl ring-1 ring-white/10 backdrop-blur-xl">
                    <header class="mb-8"><h1 class="text-4xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-purple-500">Portfolio Dashboard</h1><p class="text-slate-400 text-sm mt-2">Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC</p></header>
                    <section><h2 class="text-xl font-semibold mb-4 text-slate-300">Performance Chart (Initial: {INITIAL_CAPITAL:,.0f} CNY)</h2><div id="chart-container" class="h-[400px]"></div></section>
                    <section class="mt-8"><h2 class="text-xl font-semibold mb-4 text-slate-300">Key Performance Indicators</h2><div class="overflow-x-auto"><table class="w-full text-sm text-left"><thead class="border-b border-slate-300/20 text-slate-400"><tr><th class="py-3 px-4 font-medium">Metric</th><th class="py-3 px-4 font-medium">Portfolio</th><th class="py-3 px-4 font-medium">Benchmark (SH000300)</th></tr></thead>
                        <tbody>
                            <tr class="border-b border-slate-500/20"><td class="py-3 px-4 font-bold">Total Return</td><td class="py-3 px-4 font-mono {'text-green-400' if metrics['Total Return'] > 0 else 'text-red-400'}">{metrics['Total Return']:.2%}</td><td class="py-3 px-4 font-mono {'text-green-400' if metrics['Benchmark Total Return'] > 0 else 'text-red-400'}">{metrics['Benchmark Total Return']:.2%}</td></tr>
                            <tr class="border-b border-slate-500/20"><td class="py-3 px-4 font-bold">Annualized Return</td><td class="py-3 px-4 font-mono {'text-green-400' if metrics['Annualized Return'] > 0 else 'text-red-400'}">{metrics['Annualized Return']:.2%}</td><td class="py-3 px-4 font-mono {'text-green-400' if metrics['Benchmark Annualized Return'] > 0 else 'text-red-400'}">{metrics['Benchmark Annualized Return']:.2%}</td></tr>
                            <tr class="border-b border-slate-500/20"><td class="py-3 px-4 font-bold">Annualized Volatility</td><td class="py-3 px-4 font-mono">{metrics['Annualized Volatility']:.2%}</td><td class="py-3 px-4 font-mono">{metrics['Benchmark Annualized Volatility']:.2%}</td></tr>
                            <tr class="border-b border-slate-500/20"><td class="py-3 px-4 font-bold">Max Drawdown</td><td class="py-3 px-4 font-mono text-red-400">{metrics['Max Drawdown']:.2%}</td><td class="py-3 px-4 font-mono text-red-400">{metrics['Benchmark Max Drawdown']:.2%}</td></tr>
                            <tr class="border-b border-slate-500/20"><td class="py-3 px-4 font-bold">Sharpe Ratio</td><td class="py-3 px-4 font-mono {'text-green-400' if metrics['Sharpe Ratio'] > 0 else 'text-red-400'}">{metrics['Sharpe Ratio']:.2f}</td><td class="py-3 px-4 font-mono {'text-green-400' if metrics['Benchmark Sharpe Ratio'] > 0 else 'text-red-400'}">{metrics['Benchmark Sharpe Ratio']:.2f}</td></tr>
                            <tr><td class="py-3 px-4 font-bold">Sortino Ratio</td><td class="py-3 px-4 font-mono {'text-green-400' if metrics['Sortino Ratio'] > 0 else 'text-red-400'}">{metrics['Sortino Ratio']:.2f}</td><td class="py-3 px-4 font-mono {'text-green-400' if metrics['Benchmark Sortino Ratio'] > 0 else 'text-red-400'}">{metrics['Benchmark Sortino Ratio']:.2f}</td></tr>
                        </tbody></table></div></section>
                </div>
            </main>
            <footer class="text-center text-xs text-slate-600 pb-4">Powered by Python, QuantStats & GitHub Actions</footer>
            <script>
                const chartContainer = document.getElementById('chart-container');
                const chart = LightweightCharts.createChart(chartContainer, {{ 
                    layout: {{ backgroundColor: '#000000', textColor: '#d1d5db' }}, 
                    grid: {{ vertLines: {{ visible: false }}, horzLines: {{ visible: false }} }}, 
                    rightPriceScale: {{ scaleMargins: {{ top: 0.1, bottom: 0.1 }}, borderColor: 'rgba(255, 255, 255, 0.2)' }}, 
                    timeScale: {{ borderColor: 'rgba(255, 255, 255, 0.2)' }}, 
                    crosshair: {{ horzLine: {{ visible: false, labelVisible: false }} }} 
                }});

                // --- 最终的、经过验证的、绝对正确的修正点 ---
                const portfolioSeries = chart.addSeries(LightweightCharts.AreaSeries, {{
                    topColor: 'rgba(10, 178, 240, 0.5)', bottomColor: 'rgba(10, 178, 240, 0.01)',
                    lineColor: '#0ab2f0', lineWidth: 2, crossHairMarkerVisible: false
                }});
                const benchmarkSeries = chart.addSeries(LightweightCharts.AreaSeries, {{
                    topColor: 'rgba(168, 85, 247, 0.5)', bottomColor: 'rgba(168, 85, 247, 0.01)',
                    lineColor: '#a855f7', lineWidth: 2, crossHairMarkerVisible: false
                }});
                
                portfolioSeries.setData({str(chart_data_portfolio)});
                benchmarkSeries.setData({str(chart_data_benchmark)});

                const legend = document.createElement('div');
                legend.style = `position: absolute; left: 12px; top: 12px; z-index: 10; font-family: sans-serif;`;
                chartContainer.appendChild(legend);
                const formatPrice = price => price.toLocaleString('zh-CN', {{style: 'currency', currency: 'CNY'}});
                const setLegendText = (pPrice, bPrice, date) => {{
                    legend.innerHTML = `<div class="text-slate-400 text-sm">${{date}}</div><div class="flex items-center mt-1"><div class="w-3 h-3 rounded-full bg-[#0ab2f0] mr-2"></div><span class="text-slate-300 mr-2 text-lg">Portfolio:</span><span class="font-bold text-xl text-[#0ab2f0]">${{pPrice}}</span></div><div class="flex items-center mt-1"><div class="w-3 h-3 rounded-full bg-[#a855f7] mr-2"></div><span class="text-slate-300 mr-2 text-lg">Benchmark:</span><span class="font-bold text-xl text-[#a855f7]">${{bPrice}}</span></div>`;
                }};
                const updateLegend = param => {{
                    let pValue = 'N/A', bValue = 'N/A', date = '';
                    const validCrosshairPoint = !(param === undefined || param.time === undefined || param.point.x < 0 || param.point.y < 0);
                    if (validCrosshairPoint) {{
                        const pData = param.seriesData.get(portfolioSeries);
                        const bData = param.seriesData.get(benchmarkSeries);
                        pValue = pData ? formatPrice(pData.value) : 'N/A';
                        bValue = bData ? formatPrice(bData.value) : 'N/A';
                        date = param.time;
                    }} else {{
                        const portfolioData = {str(chart_data_portfolio)}; const benchmarkData = {str(chart_data_benchmark)};
                        if (portfolioData.length > 0) {{
                           const lastPortfolioBar = portfolioData[portfolioData.length - 1];
                           const lastBenchmarkBar = benchmarkData[benchmarkData.length - 1];
                           pValue = formatPrice(lastPortfolioBar.value);
                           bValue = formatPrice(lastBenchmarkBar.value);
                           date = lastPortfolioBar.time;
                        }}
                    }}
                    setLegendText(pValue, bValue, date);
                }};
                chart.subscribeCrosshairMove(updateLegend);
                updateLegend(undefined);
                chart.timeScale().fitContent();
                new ResizeObserver(entries => {{ if (entries.length === 0 || entries[0].target !== chartContainer) return; const newRect = entries[0].contentRect; chart.applyOptions({{ width: newRect.width, height: newRect.height }}); }}).observe(chartContainer);
            </script>
        </body></html>
        """
    with open("index.html", "w", encoding="utf-8") as f: f.write(html_content)
    print("Report 'index.html' generated successfully.")

# --- 新增: PNG 生成函数 ---
def generate_png_from_html(html_path="index.html", png_path=OUTPUT_PNG_PATH):
    """使用Playwright对本地HTML文件中的图表进行截图"""
    print(f"Starting PNG generation from {html_path}...")
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            
            # 使用 file:// 协议访问本地 HTML 文件
            absolute_html_path = os.path.abspath(html_path)
            page.goto(f'file://{absolute_html_path}')
            
            # 等待JavaScript渲染图表 (重要!)
            # 给予2秒的固定等待时间，确保图表动画和数据加载完成
            page.wait_for_timeout(2000) 
            
            # 定位到图表所在的div容器
            chart_element = page.locator('#chart-container')
            
            # 对该元素进行截图
            chart_element.screenshot(path=png_path)
            
            browser.close()
            print(f"Successfully generated PNG: {png_path}")
            return True
    except Exception as e:
        print(f"Error during PNG generation: {e}")
        return False
    
# --- 主执行逻辑 ---
if __name__ == "__main__":
    if not API_KEY or API_KEY == "YOUR_API_KEY_HERE": raise ValueError("Alpha Vantage API Key not found. Please set it as an environment variable.")
    all_tickers = TICKERS + [BENCHMARK_TICKER]; all_data = {ticker: get_data(ticker) for ticker in all_tickers}
    if any(data is None for data in all_data.values()): print("\nCritical Error: Failed to get data.")
    else:
        assets_data = [all_data[ticker] for ticker in TICKERS]; benchmark_data = all_data[BENCHMARK_TICKER]
        portfolio_returns, benchmark_returns = run_backtest(assets_data, benchmark_data)
        if portfolio_returns is None or portfolio_returns.empty:
            print("Backtest resulted in no data, likely because start date is in the future.")
            generate_html_report(is_future=True)
        else:
            generate_html_report(portfolio_returns, benchmark_returns)
            # 在生成HTML报告成功后，调用截图函数
            generate_png_from_html()