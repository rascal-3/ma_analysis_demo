import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import yfinance as yf
from datetime import datetime, timedelta
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import requests
from bs4 import BeautifulSoup
import re
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
# リクエスト間に遅延を追加
time.sleep(2)  # 2秒待機


class MAAnalysisTool:
    """
    M&A分析のための統合ツール
    
    主な機能:
    1. 財務データの異常検知
    2. 類似企業の自動抽出
    3. バリュエーション計算
    4. シナジー効果の予測
    5. ニュースセンチメント分析
    """
    
    def __init__(self):
        """初期化"""
        self.target_company = None
        self.similar_companies = None
        self.financial_data = None
        self.news_data = None
        
    def load_financial_data(self, ticker, years=5):
        """
        企業の財務データをロード - 堅牢なキャッシングとフォールバック対策あり
        """
        self.target_company = ticker
        
        # キャッシュディレクトリの確認
        cache_dir = 'data_cache'
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        
        # キャッシュファイルのパス
        cache_file = os.path.join(cache_dir, f"{ticker}_data.pkl")
        
        # キャッシュが存在し、1週間以内なら使用
        use_cache = False
        if os.path.exists(cache_file):
            file_age = time.time() - os.path.getmtime(cache_file)
            if file_age < 604800:  # 1週間 = 604800秒
                try:
                    with open(cache_file, 'rb') as f:
                        cached_data = pickle.load(f)
                        # キャッシュデータの検証
                        if (isinstance(cached_data, dict) and 
                            'historical' in cached_data and not cached_data['historical'].empty and
                            'info' in cached_data and cached_data['info']):
                            self.financial_data = cached_data
                            print(f"キャッシュからデータを読み込みました: {ticker}")
                            use_cache = True
                            return True, self.financial_data
                        else:
                            print(f"キャッシュデータが不完全です。新しいデータを取得します: {ticker}")
                except Exception as e:
                    print(f"キャッシュからの読み込みに失敗しました: {e}")
        
        # キャッシュが使えない場合は新しいデータを取得
        if not use_cache:
            print(f"新しいデータを取得します: {ticker}")
            
            # 完全に新しい財務データ辞書を初期化
            financial_data = {
                'historical': pd.DataFrame(),
                'income_statement': pd.DataFrame(),
                'balance_sheet': pd.DataFrame(),
                'cash_flow': pd.DataFrame(),
                'info': {}
            }
            
            # 段階的なデータ取得とエラーハンドリング
            try:
                # 1. まずヒストリカルデータ（最も重要）を取得
                try:
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=365*years)
                    hist_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                    if not hist_data.empty:
                        financial_data['historical'] = hist_data
                        print(f"ヒストリカルデータを取得しました: {ticker}")
                        
                        # 基本情報の一部を計算（APIに頼らない）
                        current_price = float(hist_data['Close'].iloc[-1]) if not hist_data.empty else 0
                        financial_data['info'] = {
                            'shortName': ticker,
                            'currentPrice': current_price,
                            'fiftyTwoWeekHigh': float(hist_data['High'].max()) if not hist_data.empty else 0,
                            'fiftyTwoWeekLow': float(hist_data['Low'].min()) if not hist_data.empty else 0
                        }
                except Exception as e:
                    print(f"ヒストリカルデータの取得に失敗: {e}")
                
                # 十分な間隔を空ける
                time.sleep(2)
                
                # 2. 次に基本情報の残りを取得（重要度中）
                try:
                    # yfinanceのTickerを初期化してinfo属性を取得
                    company = yf.Ticker(ticker)
                    basic_info = company.info
                    
                    # 基本情報をマージ
                    if basic_info:
                        financial_data['info'].update(basic_info)
                        print(f"基本情報を取得しました: {ticker}")
                except Exception as e:
                    print(f"基本情報の取得に失敗: {e}")
                    # 基本的なプレースホルダー情報を提供
                    basic_info = {
                        'industry': 'Unknown',
                        'sector': 'Unknown',
                        'marketCap': 0,
                        'trailingPE': 0
                    }
                    financial_data['info'].update(basic_info)
                
                # 十分な間隔を空ける
                time.sleep(2)
                
                # 3. 最後に財務諸表を取得（優先度低）- 取得できなくても他の機能が動作するようにする
                try:
                    # 各財務諸表を個別に取得
                    income_stmt = company.income_stmt
                    if not income_stmt.empty:
                        financial_data['income_statement'] = income_stmt
                        print(f"損益計算書を取得しました: {ticker}")
                except Exception as e:
                    print(f"損益計算書の取得に失敗: {e}")
                
                time.sleep(2)
                
                try:
                    balance_sheet = company.balance_sheet
                    if not balance_sheet.empty:
                        financial_data['balance_sheet'] = balance_sheet
                        print(f"貸借対照表を取得しました: {ticker}")
                except Exception as e:
                    print(f"貸借対照表の取得に失敗: {e}")
                
                time.sleep(2)
                
                try:
                    cash_flow = company.cashflow
                    if not cash_flow.empty:
                        financial_data['cash_flow'] = cash_flow
                        print(f"キャッシュフロー計算書を取得しました: {ticker}")
                except Exception as e:
                    print(f"キャッシュフロー計算書の取得に失敗: {e}")
                
                # 4. データがそれぞれ取得できたか確認し、最低限のデータでも返せるようにする
                success = not financial_data['historical'].empty and financial_data['info']  # ヒストリカルデータと基本情報があれば成功
                
                # データを設定
                self.financial_data = financial_data
                
                # キャッシュに保存 - 部分的なデータでも保存
                if success:
                    try:
                        with open(cache_file, 'wb') as f:
                            pickle.dump(financial_data, f)
                            print(f"データをキャッシュに保存しました: {ticker}")
                    except Exception as cache_error:
                        print(f"キャッシュへの保存に失敗: {cache_error}")
                
                if success:
                    return True, self.financial_data
                else:
                    return False, "主要なデータの取得に失敗しました。インターネット接続と企業ティッカーを確認してください。"
                
            except Exception as e:
                # 完全に失敗した場合のフォールバック
                return False, f"財務データの取得に失敗しました: {str(e)}"
    
    
    def detect_financial_anomalies(self):
        """
        財務データの異常値を検出
        
        Returns:
        --------
        anomalies : dict
            検出された異常値
        """
        if self.financial_data is None:
            return False, "財務データがロードされていません。load_financial_data()を先に実行してください。"
        
        try:
            # 四半期データから特徴量を作成
            hist_data = self.financial_data['historical'].copy()
            
            # マルチインデックスかどうかを確認
            if isinstance(hist_data.columns, pd.MultiIndex):
                # マルチインデックスの場合、最初のティッカーのデータを使用
                ticker = hist_data.columns.get_level_values(1)[0]
                close_col = ('Close', ticker)
                volume_col = ('Volume', ticker)
                
                # 新しいデータフレームを作成して列名を単純化
                hist_data_simple = pd.DataFrame({
                    'Close': hist_data[close_col],
                    'Volume': hist_data[volume_col]
                }, index=hist_data.index)
                hist_data = hist_data_simple
            
            # 財務指標の計算
            hist_data['Returns'] = hist_data['Close'].pct_change()
            hist_data['Volatility'] = hist_data['Returns'].rolling(window=20).std()
            hist_data['Volume_Change'] = hist_data['Volume'].pct_change()
            
            # 異常検知のための特徴量
            features = hist_data[['Returns', 'Volatility', 'Volume_Change']].dropna()
            
            # 標準化
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(features)
            
            # Isolation Forestによる異常検知
            model = IsolationForest(contamination=0.05, random_state=42)
            hist_data['Anomaly'] = pd.Series(model.fit_predict(scaled_features), index=features.index)
            hist_data['Anomaly'] = hist_data['Anomaly'].map({1: 0, -1: 1})  # 1が通常、-1が異常値
            
            # 異常値を含む日付を抽出
            anomaly_dates = hist_data[hist_data['Anomaly'] == 1].index.tolist()
            anomalies = hist_data[hist_data['Anomaly'] == 1]
            
            # 株価と異常値の可視化
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                               subplot_titles=(f"{self.target_company}の株価推移と異常検知", "取引量"))
            
            # 株価チャート
            fig.add_trace(
                go.Scatter(x=hist_data.index, y=hist_data['Close'], mode='lines', name='株価'),
                row=1, col=1
            )
            
            # 異常値
            fig.add_trace(
                go.Scatter(x=anomalies.index, y=anomalies['Close'], mode='markers', 
                          name='異常値検出', marker=dict(color='red', size=10)),
                row=1, col=1
            )
            
            # 取引量
            fig.add_trace(
                go.Bar(x=hist_data.index, y=hist_data['Volume'], name='取引量'),
                row=2, col=1
            )
            
            # レイアウトの更新
            fig.update_layout(
                height=600, 
                width=800, 
                title_text=f"{self.target_company}の財務異常検知分析"
            )
            
            # Y軸のフォーマットを設定
            fig.update_yaxes(tickformat=",.2f", tickprefix="$", row=1, col=1)
            fig.update_yaxes(tickformat=",.0f", row=2, col=1)
            
            return True, {
                'anomaly_dates': anomaly_dates,
                'anomaly_data': anomalies,
                'figure': fig,
                'data': hist_data
            }
        except Exception as e:
            return False, str(e)
    
    def find_similar_companies(self, industry=None, market_cap_range=0.5, num_companies=5):
        """
        類似企業を抽出
        
        Parameters:
        -----------
        industry : str
            業種（Noneの場合、対象企業と同じ業種）
        market_cap_range : float
            時価総額の許容範囲（例: 0.5なら±50%）
        num_companies : int
            取得する類似企業の数
            
        Returns:
        --------
        similar_companies : pd.DataFrame
            類似企業のリスト
        """
        if self.target_company is None:
            return False, "対象企業が設定されていません。load_financial_data()を先に実行してください。"
            
        try:
            # 対象企業の情報を取得
            target_info = self.financial_data['info']
            
            if industry is None:
                industry = target_info.get('industry', '')
            
            target_market_cap = target_info.get('marketCap', 0)
            target_sector = target_info.get('sector', '')
            
            # 同業種の企業をAPIから検索（簡略化のため、仮のリストを使用）
            potential_tickers = self._get_industry_tickers(industry, sector=target_sector)
            
            similar_companies = []
            
            for ticker in potential_tickers:
                if ticker == self.target_company:
                    continue
                    
                try:
                    company = yf.Ticker(ticker)
                    info = company.info
                    
                    company_market_cap = info.get('marketCap', 0)
                    
                    # 時価総額の類似性をチェック
                    if (company_market_cap >= target_market_cap * (1 - market_cap_range) and 
                        company_market_cap <= target_market_cap * (1 + market_cap_range)):
                        
                        similar_companies.append({
                            'ticker': ticker,
                            'name': info.get('shortName', ticker),
                            'market_cap': company_market_cap,
                            'industry': info.get('industry', ''),
                            'PE_ratio': info.get('trailingPE', None),
                            'EV_EBITDA': info.get('enterpriseToEbitda', None)
                        })
                        
                        if len(similar_companies) >= num_companies:
                            break
                
                except Exception as e:
                    continue
            
            self.similar_companies = pd.DataFrame(similar_companies)
            
            # 可視化
            if not self.similar_companies.empty:
                fig = px.bar(self.similar_companies, x='ticker', y='market_cap', 
                            title=f"{self.target_company}の類似企業 - 時価総額比較",
                            labels={'market_cap': '時価総額 (USD)', 'ticker': '企業'})
                
                # 対象企業を強調
                target_bar_color = ['red' if x == self.target_company else 'blue' for x in self.similar_companies['ticker']]
                fig.update_traces(marker_color=target_bar_color)
                
                return True, {'similar_companies': self.similar_companies, 'figure': fig}
            else:
                return False, "類似企業が見つかりませんでした。"
                
        except Exception as e:
            return False, str(e)
    
    def _get_industry_tickers(self, industry, sector=None):
        """
        同業種の企業ティッカーリストを取得（実際のアプリケーションではAPI連携）
        
        注: この関数はデモ用の簡略化されたものです
        """
        # デモ用の仮のティッカーリスト
        demo_tickers = {
            'Technology': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'ADBE', 'CRM', 'INTC'],
            'Healthcare': ['JNJ', 'PFE', 'MRK', 'ABBV', 'TMO', 'ABT', 'UNH', 'BMY', 'AMGN', 'MDT'],
            'Financial Services': ['JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'AXP', 'BLK', 'SPGI', 'CME'],
            'Consumer Cyclical': ['HD', 'MCD', 'NKE', 'SBUX', 'TGT', 'LOW', 'TJX', 'BKNG', 'MAR', 'DIS'],
            'Communication': ['VZ', 'T', 'CMCSA', 'NFLX', 'DIS', 'TMUS', 'ATVI', 'EA', 'DISH', 'CHTR']
        }
        
        # 対象企業を各カテゴリに追加
        for key in demo_tickers:
            if self.target_company not in demo_tickers[key]:
                demo_tickers[key].append(self.target_company)
        
        # 業種に基づいてティッカーを返す
        for key, tickers in demo_tickers.items():
            if industry and industry.lower() in key.lower() or (sector and sector.lower() in key.lower()):
                return tickers
        
        # デフォルト値として最初のリストを返す
        return list(demo_tickers.values())[0]
    
    def calculate_valuation(self, methods=['DCF', 'Multiples']):
        """
        複数の手法で企業価値を計算
        
        Parameters:
        -----------
        methods : list
            使用するバリュエーション手法のリスト
            
        Returns:
        --------
        valuation_results : dict
            バリュエーション結果
        """
        if self.financial_data is None:
            return False, "財務データがロードされていません。load_financial_data()を先に実行してください。"
            
        try:
            results = {}
            
            # ティッカー情報の取得
            info = self.financial_data['info']
            current_price = info.get('currentPrice', 0)
            shares_outstanding = info.get('sharesOutstanding', 0)
            current_market_cap = current_price * shares_outstanding
            
            # 財務データの準備
            income_stmt = self.financial_data['income_statement']
            balance_sheet = self.financial_data['balance_sheet']
            cash_flow = self.financial_data['cash_flow']
            
            # 1. DCF法によるバリュエーション
            if 'DCF' in methods:
                try:
                    # 過去のフリーキャッシュフローの取得
                    if 'FreeCashFlow' in cash_flow.index:
                        historical_fcf = cash_flow.loc['FreeCashFlow']
                    else:
                        # フリーキャッシュフローを計算 (営業CF - 設備投資)
                        operating_cash_flow = cash_flow.loc['OperatingCashFlow'] if 'OperatingCashFlow' in cash_flow.index else 0
                        capital_expenditures = cash_flow.loc['CapitalExpenditures'] if 'CapitalExpenditures' in cash_flow.index else 0
                        historical_fcf = operating_cash_flow - capital_expenditures
                    
                    # historical_fcfが整数型の場合、pandas.Seriesに変換
                    if isinstance(historical_fcf, (int, float)):
                        # 単一の値の場合、過去5年分の同じ値を持つSeriesを作成
                        historical_fcf = pd.Series([historical_fcf] * 5, index=range(5))
                        avg_growth_rate = 0.03  # デフォルト成長率を設定
                    else:
                        # 成長率の推定（過去のFCFの平均成長率）
                        fcf_growth_rates = historical_fcf.pct_change().dropna()
                        avg_growth_rate = fcf_growth_rates.mean()
                    
                    # 保守的な成長率の設定
                    if avg_growth_rate > 0.15:
                        growth_rate = 0.15  # 成長率の上限を15%に設定
                    elif avg_growth_rate < 0:
                        growth_rate = 0.03  # マイナス成長の場合、デフォルト値として3%を設定
                    else:
                        growth_rate = avg_growth_rate
                    
                    # 最新のフリーキャッシュフロー
                    latest_fcf = historical_fcf.iloc[-1]
                    
                    # DCF計算のパラメータ
                    projection_years = 5  # 予測期間
                    terminal_growth_rate = 0.02  # 永続成長率
                    discount_rate = 0.1  # 割引率（WACC）
                    
                    # 将来キャッシュフローの予測と現在価値計算
                    projected_fcf = []
                    present_values = []
                    
                    for year in range(1, projection_years + 1):
                        # t年後のフリーキャッシュフロー
                        fcf_t = latest_fcf * (1 + growth_rate) ** year
                        # 割引係数
                        discount_factor = (1 + discount_rate) ** year
                        # 現在価値
                        present_value = fcf_t / discount_factor
                        
                        projected_fcf.append(fcf_t)
                        present_values.append(present_value)
                    
                    # 継続価値（ターミナルバリュー）の計算
                    terminal_fcf = projected_fcf[-1] * (1 + terminal_growth_rate)
                    terminal_value = terminal_fcf / (discount_rate - terminal_growth_rate)
                    terminal_value_present = terminal_value / (1 + discount_rate) ** projection_years
                    
                    # 企業価値 = 予測期間の現在価値の合計 + 継続価値の現在価値
                    enterprise_value = sum(present_values) + terminal_value_present
                    
                    # 純有利子負債の計算
                    total_debt = balance_sheet.loc['TotalDebt'].iloc[-1] if 'TotalDebt' in balance_sheet.index else 0
                    cash_and_equiv = balance_sheet.loc['CashAndCashEquivalents'].iloc[-1] if 'CashAndCashEquivalents' in balance_sheet.index else 0
                    net_debt = total_debt - cash_and_equiv
                    
                    # 株主価値 = 企業価値 - 純有利子負債
                    equity_value = enterprise_value - net_debt
                    
                    # 1株あたり価値
                    per_share_value = equity_value / shares_outstanding if shares_outstanding > 0 else 0
                    
                    # 結果の格納
                    results['DCF'] = {
                        'enterprise_value': enterprise_value,
                        'equity_value': equity_value,
                        'per_share_value': per_share_value,
                        'current_price': current_price,
                        'upside_potential': (per_share_value / current_price - 1) * 100 if current_price > 0 else 0,
                        'parameters': {
                            'growth_rate': growth_rate,
                            'discount_rate': discount_rate,
                            'terminal_growth_rate': terminal_growth_rate,
                            'projection_years': projection_years
                        }
                    }
                    
                except Exception as e:
                    results['DCF'] = {'error': str(e)}
            
            # 2. マルチプル法によるバリュエーション
            if 'Multiples' in methods:
                try:
                    # 類似企業データの取得
                    if self.similar_companies is None:
                        success, result = self.find_similar_companies()
                        if not success:
                            results['Multiples'] = {'error': '類似企業データを取得できませんでした: ' + result}
                            return True, {'results': results}
                    
                    if self.similar_companies.empty:
                        results['Multiples'] = {'error': '類似企業データがありません'}
                    else:
                        # EV/EBITDA倍率の計算
                        peer_ev_ebitda = self.similar_companies['EV_EBITDA'].dropna().median()
                        
                        # 対象企業のEBITDAの取得
                        if 'EBITDA' in income_stmt.index:
                            ebitda = income_stmt.loc['EBITDA'].iloc[-1]
                        else:
                            # EBITDAの計算 (EBIT + 減価償却費)
                            ebit = income_stmt.loc['EBIT'].iloc[-1] if 'EBIT' in income_stmt.index else 0
                            depreciation = income_stmt.loc['DepreciationAndAmortization'].iloc[-1] if 'DepreciationAndAmortization' in income_stmt.index else 0
                            ebitda = ebit + depreciation
                        
                        # EV/EBITDA倍率による企業価値
                        ev_ebitda_value = ebitda * peer_ev_ebitda
                        
                        # 純有利子負債の計算
                        total_debt = balance_sheet.loc['TotalDebt'].iloc[-1] if 'TotalDebt' in balance_sheet.index else 0
                        cash_and_equiv = balance_sheet.loc['CashAndCashEquivalents'].iloc[-1] if 'CashAndCashEquivalents' in balance_sheet.index else 0
                        net_debt = total_debt - cash_and_equiv
                        
                        # 株主価値 = 企業価値 - 純有利子負債
                        equity_value_ebitda = ev_ebitda_value - net_debt
                        
                        # 1株あたり価値
                        per_share_value_ebitda = equity_value_ebitda / shares_outstanding if shares_outstanding > 0 else 0
                        
                        # P/E倍率の計算
                        peer_pe = self.similar_companies['PE_ratio'].dropna().median()
                        
                        # 対象企業のEPSの取得
                        net_income = income_stmt.loc['NetIncome'].iloc[-1] if 'NetIncome' in income_stmt.index else 0
                        eps = net_income / shares_outstanding if shares_outstanding > 0 else 0
                        
                        # P/E倍率による株主価値
                        pe_value = eps * peer_pe
                        
                        # 結果の格納
                        results['Multiples'] = {
                            'EV_EBITDA': {
                                'multiple': peer_ev_ebitda,
                                'enterprise_value': ev_ebitda_value,
                                'equity_value': equity_value_ebitda,
                                'per_share_value': per_share_value_ebitda,
                                'upside_potential': (per_share_value_ebitda / current_price - 1) * 100 if current_price > 0 else 0
                            },
                            'PE': {
                                'multiple': peer_pe,
                                'per_share_value': pe_value,
                                'upside_potential': (pe_value / current_price - 1) * 100 if current_price > 0 else 0
                            }
                        }
                        
                except Exception as e:
                    results['Multiples'] = {'error': str(e)}
            
            # バリュエーション結果の可視化
            if len(results) > 0:
                # 結果をまとめる
                valuation_summary = []
                
                if 'DCF' in results and 'per_share_value' in results['DCF']:
                    valuation_summary.append({
                        'Method': 'DCF Method',
                        'Value': results['DCF']['per_share_value'],
                        'Upside': results['DCF']['upside_potential']
                    })
                
                if 'Multiples' in results and 'error' not in results['Multiples']:
                    if 'EV_EBITDA' in results['Multiples']:
                        valuation_summary.append({
                            'Method': 'EV/EBITDA Multiple',
                            'Value': results['Multiples']['EV_EBITDA']['per_share_value'],
                            'Upside': results['Multiples']['EV_EBITDA']['upside_potential']
                        })
                    
                    if 'PE' in results['Multiples']:
                        valuation_summary.append({
                            'Method': 'P/E Multiple',
                            'Value': results['Multiples']['PE']['per_share_value'],
                            'Upside': results['Multiples']['PE']['upside_potential']
                        })
                
                # 現在の株価を追加
                valuation_summary.append({
                    'Method': 'Current Market Price',
                    'Value': current_price,
                    'Upside': 0
                })
                
                # データフレームに変換
                df_summary = pd.DataFrame(valuation_summary)
                
                # バリュエーション方法ごとの株価の可視化
                fig = px.bar(df_summary, x='Method', y='Value', 
                            title=f"{self.target_company}のバリュエーション比較",
                            labels={'Value': '株価 (USD)', 'Method': 'バリュエーション手法'},
                            text='Value')
                
                fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
                fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
                
                # 現在の株価を強調
                market_price_color = ['blue' if x != 'Current Market Price' else 'red' for x in df_summary['Method']]
                fig.update_traces(marker_color=market_price_color)
                
                # アップサイドポテンシャルの可視化
                upside_data = df_summary[df_summary['Method'] != 'Current Market Price']
                
                if not upside_data.empty:
                    fig2 = px.bar(upside_data, x='Method', y='Upside',
                                 title=f"{self.target_company}のアップサイドポテンシャル",
                                 labels={'Upside': 'アップサイド (%)', 'Method': 'バリュエーション手法'},
                                 text='Upside')
                    
                    fig2.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                    fig2.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
                    
                    # 正負によって色を変える
                    bar_colors = ['green' if x > 0 else 'red' for x in upside_data['Upside']]
                    fig2.update_traces(marker_color=bar_colors)
                    
                    return True, {'results': results, 'summary': df_summary, 'figure': fig, 'figure_upside': fig2}
                
                return True, {'results': results, 'summary': df_summary, 'figure': fig}
            
            return True, {'results': results}
            
        except Exception as e:
            return False, str(e)
    
    def analyze_news_sentiment(self, period_days=30):
        """
        企業に関するニュースのセンチメント分析を行います
        
        Parameters:
        -----------
        period_days : int
            分析する期間（日数）
            
        Returns:
        --------
        sentiment_results : dict
            センチメント分析結果
        """
        if self.target_company is None:
            return False, "対象企業が設定されていません。load_financial_data()を先に実行してください。"
        
        try:
            # NLTKの感情分析ツールをダウンロード（初回のみ）
            try:
                nltk.data.find('vader_lexicon')
            except LookupError:
                nltk.download('vader_lexicon')
            
            # センチメント分析器の初期化
            sia = SentimentIntensityAnalyzer()
            
            # ニュース記事を取得（実際のアプリケーションではニュースAPIを使用）
            news_articles = self._fetch_company_news(period_days)
            
            # 各記事のセンチメントを分析
            for article in news_articles:
                sentiment = sia.polarity_scores(article['title'] + ' ' + article['snippet'])
                article['sentiment'] = sentiment
                article['sentiment_compound'] = sentiment['compound']
            
            # 日付ごとのセンチメントを集計
            sentiment_by_date = {}
            
            for article in news_articles:
                date = article['date'].strftime('%Y-%m-%d')
                if date not in sentiment_by_date:
                    sentiment_by_date[date] = []
                
                sentiment_by_date[date].append(article['sentiment_compound'])
            
            # 日付ごとの平均センチメントを計算
            daily_sentiment = []
            
            for date, scores in sentiment_by_date.items():
                daily_sentiment.append({
                    'date': date,
                    'avg_sentiment': sum(scores) / len(scores),
                    'article_count': len(scores)
                })
            
            # 日付順にソート
            daily_sentiment = sorted(daily_sentiment, key=lambda x: x['date'])
            
            # 株価データと結合
            if self.financial_data is not None:
                historical_data = self.financial_data['historical'].copy()
                sentiment_df = pd.DataFrame(daily_sentiment)
                sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
                sentiment_df.set_index('date', inplace=True)
                
                # 日付インデックスを合わせる
                start_date = max(historical_data.index.min(), sentiment_df.index.min())
                end_date = min(historical_data.index.max(), sentiment_df.index.max())
                historical_subset = historical_data.loc[start_date:end_date]
                sentiment_subset = sentiment_df.loc[start_date:end_date]
                
                # 可視化
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                   subplot_titles=(f"{self.target_company}の株価とニュースセンチメントの関係",
                                                 "ニュースの感情スコア"))
                
                # 株価チャート
                fig.add_trace(
                    go.Scatter(x=historical_subset.index, y=historical_subset['Close'], mode='lines', name='株価'),
                    row=1, col=1
                )
                
                # センチメントスコア
                fig.add_trace(
                    go.Bar(x=sentiment_subset.index, y=sentiment_subset['avg_sentiment'], name='センチメントスコア',
                          marker=dict(color=sentiment_subset['avg_sentiment'].apply(
                              lambda x: 'green' if x > 0.2 else 'red' if x < -0.2 else 'gray'))),
                    row=2, col=1
                )
                
                fig.update_layout(height=600, width=800)
            
            # ニュースのセンチメント傾向の概要
            positive_articles = [a for a in news_articles if a['sentiment_compound'] > 0.2]
            negative_articles = [a for a in news_articles if a['sentiment_compound'] < -0.2]
            neutral_articles = [a for a in news_articles if -0.2 <= a['sentiment_compound'] <= 0.2]
            
            sentiment_summary = {
                'positive_count': len(positive_articles),
                'negative_count': len(negative_articles),
                'neutral_count': len(neutral_articles),
                'total_articles': len(news_articles),
                'avg_sentiment': sum(a['sentiment_compound'] for a in news_articles) / len(news_articles) if news_articles else 0,
                'sentiment_trend': daily_sentiment,
                'top_positive': sorted(positive_articles, key=lambda x: x['sentiment_compound'], reverse=True)[:3] if positive_articles else [],
                'top_negative': sorted(negative_articles, key=lambda x: x['sentiment_compound'])[:3] if negative_articles else []
            }
            
            # 感情分布の可視化
            sentiment_counts = [
                sentiment_summary['positive_count'],
                sentiment_summary['neutral_count'],
                sentiment_summary['negative_count']
            ]
            
            labels = ['ポジティブ', '中立', 'ネガティブ']
            colors = ['green', 'gray', 'red']
            
            fig_pie = go.Figure(data=[go.Pie(labels=labels, values=sentiment_counts, marker=dict(colors=colors))])
            fig_pie.update_layout(title=f"{self.target_company}に関するニュースセンチメントの分布")
            
            self.news_data = {
                'articles': news_articles,
                'summary': sentiment_summary
            }
            
            return True, {'summary': sentiment_summary, 'figure': fig, 'figure_pie': fig_pie, 'articles': news_articles}
            
        except Exception as e:
            return False, str(e)
    
    def _fetch_company_news(self, period_days=30):
        """
        企業に関するニュース記事を取得（デモ用の簡易実装）
        実際のアプリケーションではNewsAPI等の外部APIを使用
        
        Parameters:
        -----------
        period_days : int
            取得する期間（日数）
            
        Returns:
        --------
        news_articles : list
            ニュース記事のリスト
        """
        # Yahoo Financeからニュースを取得する試み
        try:
            company = yf.Ticker(self.target_company)
            news = company.news
            
            if news and len(news) > 0:
                articles = []
                for item in news:
                    article = {
                        'title': item.get('title', ''),
                        'date': datetime.fromtimestamp(item.get('providerPublishTime', time.time())),
                        'snippet': item.get('summary', ''),
                        'source': item.get('publisher', 'Yahoo Finance')
                    }
                    articles.append(article)
                return articles
        except:
            pass  # Yahoo Finance APIからニュースを取得できない場合はデモデータを使用
        
        # デモ用の架空のニュースデータ
        return self._generate_demo_news(period_days)
    
    def _generate_demo_news(self, period_days):
        """デモ用の架空のニュースデータを生成"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=period_days)
        
        # 企業名
        company_name = yf.Ticker(self.target_company).info.get('shortName', self.target_company)
        
        # ポジティブなニュースのテンプレート
        positive_templates = [
            "{company}、四半期決算で市場予想を上回る",
            "{company}、新製品の販売好調で株価上昇",
            "{company}、戦略的提携を発表し投資家から好評",
            "{company}の新CEOの戦略が高く評価される",
            "{company}、コスト削減計画が進展し利益率改善",
            "アナリスト、{company}の株価目標を引き上げ",
            "{company}、主力事業が好調で通期見通しを上方修正",
            "{company}、技術革新で業界をリード",
            "{company}、拡大するマーケットシェアでポジション強化",
            "{company}、サステナビリティ目標で高評価を獲得"
        ]
        
        # ネガティブなニュースのテンプレート
        negative_templates = [
            "{company}、業績予想を下方修正",
            "{company}の四半期決算、市場予想を下回る",
            "{company}、競争激化で市場シェア低下",
            "{company}、主力製品の問題で株価下落",
            "規制当局、{company}に調査開始",
            "{company}、リストラ計画を発表",
            "アナリスト、{company}の株価目標を引き下げ",
            "{company}のCEO辞任、後継者問題で不安定化",
            "{company}、特許侵害で訴訟に直面",
            "{company}、サプライチェーン問題で生産減少"
        ]
        
        # 中立的なニュースのテンプレート
        neutral_templates = [
            "{company}、新たな市場進出を検討",
            "{company}、組織再編を発表",
            "{company}、新たな取締役を任命",
            "{company}、業界カンファレンスで新戦略を発表",
            "{company}、持続可能性レポートを公開",
            "{company}、デジタル変革の進捗状況を報告",
            "{company}、事業多角化の計画を検討中",
            "{company}の株主総会、主要議案を承認",
            "{company}、研究開発投資の方針を更新",
            "{company}、業界トレンドへの対応を説明"
        ]
        
        # ニュース記事の生成
        news_articles = []
        current_date = start_date
        
        while current_date <= end_date:
            # この日のニュース記事数（0-3）
            article_count = np.random.randint(0, 4)
            
            for _ in range(article_count):
                # ニュースの種類（ポジティブ/ネガティブ/中立）をランダムに選択
                news_type = np.random.choice(['positive', 'negative', 'neutral'], 
                                          p=[0.4, 0.3, 0.3])  # 確率分布
                
                if news_type == 'positive':
                    template = np.random.choice(positive_templates)
                    snippet = "業界アナリストによると、この発表は同社の長期的な成長戦略に合致しており、投資家から前向きな反応を得ています。"
                elif news_type == 'negative':
                    template = np.random.choice(negative_templates)
                    snippet = "この発表を受けて株価は下落し、投資家の間で懸念が広がっています。アナリストは今後の見通しについて慎重な姿勢を示しています。"
                else:  # neutral
                    template = np.random.choice(neutral_templates)
                    snippet = "市場関係者はこの動きを注視しており、今後の展開が期待されています。"
                
                # 記事タイトルの生成
                title = template.format(company=company_name)
                
                # 記事時間（その日の9時〜18時のランダムな時間）
                hour = np.random.randint(9, 18)
                minute = np.random.randint(0, 60)
                article_datetime = current_date.replace(hour=hour, minute=minute)
                
                news_articles.append({
                    'title': title,
                    'date': article_datetime,
                    'snippet': snippet,
                    'source': np.random.choice(['Financial Times', 'Wall Street Journal', 'Bloomberg', 'Reuters', 'CNBC'])
                })
            
            # 次の日へ
            current_date += timedelta(days=1)
        
        return news_articles
    
    def estimate_synergy(self, target_ticker, synergy_types=['Revenue', 'Cost', 'Financial']):
        """
        M&Aによるシナジー効果の推定
        
        Parameters:
        -----------
        target_ticker : str
            買収対象企業のティッカーシンボル
        synergy_types : list
            分析するシナジーのタイプ
            
        Returns:
        --------
        synergy_results : dict
            シナジー分析結果
        """
        if self.financial_data is None:
            return False, "財務データがロードされていません。load_financial_data()を先に実行してください。"
        
        # 買収対象企業の財務データを取得
        try:
            target_company = yf.Ticker(target_ticker)
            
            target_income = target_company.income_stmt
            target_balance = target_company.balance_sheet
            
            # 両社の財務データ
            acquirer_income = self.financial_data['income_statement']
            acquirer_balance = self.financial_data['balance_sheet']
            
            # 結果格納用
            synergy_results = {}
            all_figures = []
            
            # 1. 収益シナジー
            if 'Revenue' in synergy_types:
                # 両社の売上
                acquirer_revenue = acquirer_income.loc['TotalRevenue'].iloc[-1] if 'TotalRevenue' in acquirer_income.index else 0
                target_revenue = target_income.loc['TotalRevenue'].iloc[-1] if 'TotalRevenue' in target_income.index else 0
                
                # 収益シナジーの仮定（例：クロスセルにより5-15%の収益向上）
                revenue_synergy_low = (acquirer_revenue + target_revenue) * 0.05
                revenue_synergy_mid = (acquirer_revenue + target_revenue) * 0.1
                revenue_synergy_high = (acquirer_revenue + target_revenue) * 0.15
                
                synergy_results['Revenue'] = {
                    'acquirer_revenue': acquirer_revenue,
                    'target_revenue': target_revenue,
                    'combined_revenue': acquirer_revenue + target_revenue,
                    'synergy_low': revenue_synergy_low,
                    'synergy_mid': revenue_synergy_mid,
                    'synergy_high': revenue_synergy_high
                }
            
            # 2. コストシナジー
            if 'Cost' in synergy_types:
                # 両社の営業費用
                if 'OperatingExpense' in acquirer_income.index:
                    acquirer_opex = acquirer_income.loc['OperatingExpense'].iloc[-1]
                else:
                    # 営業費用 = 売上 - 営業利益
                    acquirer_revenue = acquirer_income.loc['TotalRevenue'].iloc[-1] if 'TotalRevenue' in acquirer_income.index else 0
                    acquirer_operating_income = acquirer_income.loc['OperatingIncome'].iloc[-1] if 'OperatingIncome' in acquirer_income.index else 0
                    acquirer_opex = acquirer_revenue - acquirer_operating_income
                
                if 'OperatingExpense' in target_income.index:
                    target_opex = target_income.loc['OperatingExpense'].iloc[-1]
                else:
                    # 営業費用 = 売上 - 営業利益
                    target_revenue = target_income.loc['TotalRevenue'].iloc[-1] if 'TotalRevenue' in target_income.index else 0
                    target_operating_income = target_income.loc['OperatingIncome'].iloc[-1] if 'OperatingIncome' in target_income.index else 0
                    target_opex = target_revenue - target_operating_income
                
                # コストシナジーの仮定（例：重複コストの削減により10-30%のコスト削減）
                cost_synergy_low = (acquirer_opex + target_opex) * 0.1
                cost_synergy_mid = (acquirer_opex + target_opex) * 0.2
                cost_synergy_high = (acquirer_opex + target_opex) * 0.3
                
                synergy_results['Cost'] = {
                    'acquirer_opex': acquirer_opex,
                    'target_opex': target_opex,
                    'combined_opex': acquirer_opex + target_opex,
                    'synergy_low': cost_synergy_low,
                    'synergy_mid': cost_synergy_mid,
                    'synergy_high': cost_synergy_high
                }
            
            # 3. 財務シナジー
            if 'Financial' in synergy_types:
                # 両社の加重平均資本コスト（WACC）を推定
                # 実際のアプリケーションではより精緻な計算を行う
                estimated_acquirer_wacc = 0.09  # 仮の値
                estimated_target_wacc = 0.11  # 仮の値
                
                # 両社の企業価値
                acquirer_market_cap = yf.Ticker(self.target_company).info.get('marketCap', 0)
                target_market_cap = target_company.info.get('marketCap', 0)
                
                # 統合後のWACC低下の想定（例：0.5-1.5%ポイントの低下）
                combined_wacc_low = (estimated_acquirer_wacc * acquirer_market_cap + estimated_target_wacc * target_market_cap) / (acquirer_market_cap + target_market_cap) - 0.005
                combined_wacc_mid = (estimated_acquirer_wacc * acquirer_market_cap + estimated_target_wacc * target_market_cap) / (acquirer_market_cap + target_market_cap) - 0.01
                combined_wacc_high = (estimated_acquirer_wacc * acquirer_market_cap + estimated_target_wacc * target_market_cap) / (acquirer_market_cap + target_market_cap) - 0.015
                
                # 両社の時価総額の合計
                combined_market_cap = acquirer_market_cap + target_market_cap
                
                # 財務シナジーの価値（WACC低下による企業価値向上）
                financial_synergy_low = combined_market_cap * (0.005 / combined_wacc_low)
                financial_synergy_mid = combined_market_cap * (0.01 / combined_wacc_mid)
                financial_synergy_high = combined_market_cap * (0.015 / combined_wacc_high)
                
                synergy_results['Financial'] = {
                    'acquirer_wacc': estimated_acquirer_wacc,
                    'target_wacc': estimated_target_wacc,
                    'combined_wacc_low': combined_wacc_low,
                    'combined_wacc_mid': combined_wacc_mid,
                    'combined_wacc_high': combined_wacc_high,
                    'synergy_low': financial_synergy_low,
                    'synergy_mid': financial_synergy_mid,
                    'synergy_high': financial_synergy_high
                }
            
            # 総シナジー効果
            total_synergy_low = 0
            total_synergy_mid = 0
            total_synergy_high = 0
            
            if 'Revenue' in synergy_results:
                total_synergy_low += synergy_results['Revenue']['synergy_low']
                total_synergy_mid += synergy_results['Revenue']['synergy_mid']
                total_synergy_high += synergy_results['Revenue']['synergy_high']
            
            if 'Cost' in synergy_results:
                total_synergy_low += synergy_results['Cost']['synergy_low']
                total_synergy_mid += synergy_results['Cost']['synergy_mid']
                total_synergy_high += synergy_results['Cost']['synergy_high']
            
            if 'Financial' in synergy_results:
                total_synergy_low += synergy_results['Financial']['synergy_low']
                total_synergy_mid += synergy_results['Financial']['synergy_mid']
                total_synergy_high += synergy_results['Financial']['synergy_high']
            
            synergy_results['Total'] = {
                'synergy_low': total_synergy_low,
                'synergy_mid': total_synergy_mid,
                'synergy_high': total_synergy_high
            }
            
            # シナジー効果の可視化
            # シナジータイプごとの効果を集計
            synergy_data = []
            
            if 'Revenue' in synergy_results:
                synergy_data.append({
                    'Type': 'Revenue Synergy',
                    'Low': synergy_results['Revenue']['synergy_low'],
                    'Mid': synergy_results['Revenue']['synergy_mid'],
                    'High': synergy_results['Revenue']['synergy_high']
                })
            
            if 'Cost' in synergy_results:
                synergy_data.append({
                    'Type': 'Cost Synergy',
                    'Low': synergy_results['Cost']['synergy_low'],
                    'Mid': synergy_results['Cost']['synergy_mid'],
                    'High': synergy_results['Cost']['synergy_high']
                })
            
            if 'Financial' in synergy_results:
                synergy_data.append({
                    'Type': 'Financial Synergy',
                    'Low': synergy_results['Financial']['synergy_low'],
                    'Mid': synergy_results['Financial']['synergy_mid'],
                    'High': synergy_results['Financial']['synergy_high']
                })
            
            synergy_data.append({
                'Type': 'Total Synergy',
                'Low': total_synergy_low,
                'Mid': total_synergy_mid,
                'High': total_synergy_high
            })
            
            # データフレームに変換
            df_synergy = pd.DataFrame(synergy_data)
            
            # シナジー効果の棒グラフ
            fig = go.Figure()
            
            # 低位シナリオ
            fig.add_trace(go.Bar(
                name='Low Estimate',
                x=df_synergy['Type'],
                y=df_synergy['Low'],
                marker_color='lightblue'
            ))
            
            # 中位シナリオ
            fig.add_trace(go.Bar(
                name='Mid Estimate',
                x=df_synergy['Type'],
                y=df_synergy['Mid'],
                marker_color='royalblue'
            ))
            
            # 高位シナリオ
            fig.add_trace(go.Bar(
                name='High Estimate',
                x=df_synergy['Type'],
                y=df_synergy['High'],
                marker_color='darkblue'
            ))
            
            # グラフのレイアウト設定
            fig.update_layout(
                title=f"{self.target_company}による{target_ticker}買収のシナジー効果推定",
                xaxis_title='シナジータイプ',
                yaxis_title='シナジー価値 (USD)',
                barmode='group'
            )
            
            all_figures.append(fig)
            
            # シナジーによる買収価値の増加を可視化
            target_market_cap_with_premium = target_market_cap * 1.3  # 30%プレミアムと仮定
            
            acquisition_data = [
                {'Category': 'Target Market Cap', 'Value': target_market_cap},
                {'Category': 'Target with Premium', 'Value': target_market_cap_with_premium},
                {'Category': 'Target with Premium + Synergy (Low)', 'Value': target_market_cap_with_premium + total_synergy_low},
                {'Category': 'Target with Premium + Synergy (Mid)', 'Value': target_market_cap_with_premium + total_synergy_mid},
                {'Category': 'Target with Premium + Synergy (High)', 'Value': target_market_cap_with_premium + total_synergy_high}
            ]
            
            df_acquisition = pd.DataFrame(acquisition_data)
            
            # ウォーターフォールチャート
            fig2 = px.bar(df_acquisition, x='Category', y='Value',
                        title=f"{target_ticker}の買収価値分析",
                        labels={'Value': '価値 (USD)', 'Category': 'カテゴリ'})
            
            fig2.update_layout(showlegend=False)
            all_figures.append(fig2)
            
            return True, {'results': synergy_results, 'figures': all_figures, 'synergy_data': synergy_data, 'acquisition_data': df_acquisition}
            
        except Exception as e:
            return False, f"シナジー推定中にエラーが発生しました: {e}"
        