import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import yfinance as yf
import base64
from io import BytesIO
import time

# カスタムモジュールのインポート
from ma_analysis_tool import MAAnalysisTool

# アプリのタイトルとスタイル設定
st.set_page_config(page_title="M&A分析ツール", layout="wide", initial_sidebar_state="expanded")

# CSSスタイル
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.8rem;
        color: #2563EB;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .subsection-header {
        font-size: 1.4rem;
        color: #3B82F6;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .info-box {
        background-color: #EFF6FF;
        border-left: 5px solid #3B82F6;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #FEF2F2;
        border-left: 5px solid #EF4444;
        padding: 1rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #ECFDF5;
        border-left: 5px solid #10B981;
        padding: 1rem;
        margin: 1rem 0;
    }
    .plotly-chart {
        width: 100%;
        height: 500px;
    }
</style>
""", unsafe_allow_html=True)

# KPMGのロゴやカラーを模したスタイル
st.markdown("""
<div style="background-color: #003A70; padding: 10px; border-radius: 5px; margin-bottom: 20px;">
    <h1 style="color: white; text-align: center;">M&A アナリティクス プラットフォーム</h1>
    <p style="color: white; text-align: center;">Science & Technology Group Demo</p>
</div>
""", unsafe_allow_html=True)

# 初期化
if 'ma_tool' not in st.session_state:
    st.session_state.ma_tool = MAAnalysisTool()

if 'target_company' not in st.session_state:
    st.session_state.target_company = None

if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {}

# サイドバー
st.sidebar.title("分析設定")

# 企業の選択
ticker_input = st.sidebar.text_input("企業ティッカーを入力", "AAPL", help="例: AAPL (Apple Inc.), MSFT (Microsoft Corp.)")
years = st.sidebar.slider("分析期間（年）", 1, 10, 5)

# 分析の実行ボタン
if st.sidebar.button("企業データをロード"):
    with st.spinner("企業データを取得中..."):
        success, result = st.session_state.ma_tool.load_financial_data(ticker_input, years)
        
        if success:
            st.session_state.target_company = ticker_input
            st.session_state.analysis_results = {}
            st.success(f"{ticker_input}の財務データを{years}年分ロードしました。")
        else:
            st.error(f"企業データの取得に失敗しました。エラー: {result}")

# タブを作成
tab1, tab2, tab3, tab4, tab5 = st.tabs(["財務概要", "異常検知", "バリュエーション", "類似企業", "M&Aシミュレーション"])

# タブ1: 財務概要
with tab1:
    if st.session_state.target_company:
        st.markdown(f"<h2 class='section-header'>{st.session_state.target_company}の財務概要</h2>", unsafe_allow_html=True)
        
        # # 企業情報の取得
        # company = yf.Ticker(st.session_state.target_company)
        # info = company.info
        # ✅ 良い方法（MAAnalysisToolクラスを通してデータを取得）
        info = {}  # デフォルト値を設定
        if st.session_state.target_company:
            # 財務データはすでにロード済みなので再度API呼び出しをしない
            if 'info' in st.session_state.ma_tool.financial_data and st.session_state.ma_tool.financial_data['info']:
                info = st.session_state.ma_tool.financial_data['info']
            else:
                # 財務データが不完全な場合、再度取得を試みる
                with st.spinner("企業データを再取得中..."):
                    success, result = st.session_state.ma_tool.load_financial_data(st.session_state.target_company, years)
                    if success and 'info' in st.session_state.ma_tool.financial_data:
                        info = st.session_state.ma_tool.financial_data['info']
                    else:
                        st.warning("企業データの取得に問題があります。サイドバーから「企業データをロード」ボタンを再度クリックしてください。")
        
        # 概要情報
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<h3 class='subsection-header'>企業概要</h3>", unsafe_allow_html=True)
            
            # 会社名と業種
            st.markdown(f"**会社名:** {info.get('shortName', 'N/A')}")
            st.markdown(f"**ティッカー:** {st.session_state.target_company}")
            st.markdown(f"**業種:** {info.get('industry', 'N/A')}")
            st.markdown(f"**セクター:** {info.get('sector', 'N/A')}")
            st.markdown(f"**国:** {info.get('country', 'N/A')}")
            # 条件分岐による解決策
            employees = info.get('fullTimeEmployees')
            if isinstance(employees, (int, float)):
                st.markdown(f"**従業員数:** {employees:,}")
            else:
                st.markdown(f"**従業員数:** N/A")
            
            # ビジネス概要
            st.markdown("<h4>ビジネス概要</h4>", unsafe_allow_html=True)
            st.markdown(f"{info.get('longBusinessSummary', 'N/A')}")
            
        with col2:
            st.markdown("<h3 class='subsection-header'>市場データ</h3>", unsafe_allow_html=True)
            
            # 株価と時価総額
            current_price = info.get('currentPrice', 0)
            market_cap = info.get('marketCap', 0)
            
            st.markdown(f"**現在株価:** ${current_price:,.2f}")
            st.markdown(f"**時価総額:** ${market_cap:,.0f}")
            st.markdown(f"**52週高値:** ${info.get('fiftyTwoWeekHigh', 0):,.2f}")
            st.markdown(f"**52週安値:** ${info.get('fiftyTwoWeekLow', 0):,.2f}")
            
            # 財務指標
            st.markdown("<h4>主要財務指標</h4>", unsafe_allow_html=True)
            st.markdown(f"**P/E倍率:** {info.get('trailingPE', 'N/A')}")
            st.markdown(f"**EPS (TTM):** ${info.get('trailingEps', 0):,.2f}")
            st.markdown(f"**売上高 (TTM):** ${info.get('totalRevenue', 0):,.0f}")
            st.markdown(f"**営業利益率:** {info.get('operatingMargins', 0) * 100:.2f}%")
            
            # 配当関連指標の表示を修正
            dividend_yield = info.get('dividendYield', 0)
            trailing_yield = info.get('trailingAnnualDividendYield', 0)
            
            # dividendYieldの値が1未満かどうかをチェック
            if dividend_yield < 1:
                # 小数形式（例：0.0047）なので、100を掛けてパーセンテージ表示にする
                formatted_yield = dividend_yield * 100
            else:
                # すでにパーセンテージ形式（例：48）の場合はそのまま表示
                formatted_yield = dividend_yield
                
            # 実際の配当利回りを表示（trailingAnnualDividendYieldを優先）
            if trailing_yield > 0:
                st.markdown(f"**配当利回り:** {trailing_yield * 100:.2f}%")
            else:
                st.markdown(f"**配当利回り:** {formatted_yield:.2f}%")
            
            # 配当性向も表示
            payout_ratio = info.get('payoutRatio', 0)
            # payoutRatioも小数で表現されているので、100を掛けてパーセンテージ表示にする
            st.markdown(f"**配当性向:** {payout_ratio * 100:.2f}%")
            
            # デバッグ情報（開発完了後に削除可能）
            if st.checkbox("デバッグ情報を表示"):
                st.write("### デバッグ情報")
                st.write("#### 配当関連情報")
                for key, value in info.items():
                    if key in ['dividendYield', 'payoutRatio', 'dividendRate', 'trailingAnnualDividendRate', 'trailingAnnualDividendYield']:
                        st.write(f"{key}: {value}")
        
        # 株価チャート
        st.markdown("<h3 class='subsection-header'>株価推移</h3>", unsafe_allow_html=True)
        
        if 'historical' in st.session_state.ma_tool.financial_data:
            hist_data = st.session_state.ma_tool.financial_data['historical'].copy()
            
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
            else:
                # 通常のインデックスの場合は何もしない
                # すでに 'Close' と 'Volume' という列名があるはず
                pass
            
            # 株価チャート
            fig = px.line(hist_data, x=hist_data.index, y='Close', title=f"{st.session_state.target_company}の株価推移")
            fig.update_layout(
                xaxis_title='日付', 
                yaxis_title='株価 (USD)',
                yaxis=dict(tickformat=",.2f", tickprefix="$")
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # 出来高チャート
            fig2 = px.bar(hist_data, x=hist_data.index, y='Volume', title=f"{st.session_state.target_company}の出来高推移")
            fig2.update_layout(
                xaxis_title='日付', 
                yaxis_title='出来高',
                yaxis=dict(tickformat=",.0f")
            )
            st.plotly_chart(fig2, use_container_width=True)
        
        # 財務諸表サマリー
        st.markdown("<h3 class='subsection-header'>財務諸表サマリー</h3>", unsafe_allow_html=True)
        
        fin_tabs = st.tabs(["損益計算書", "貸借対照表", "キャッシュフロー"])
        
        with fin_tabs[0]:
            if 'income_statement' in st.session_state.ma_tool.financial_data:
                income_stmt = st.session_state.ma_tool.financial_data['income_statement']
                
                # 数値を3桁区切りでフォーマット
                income_stmt_formatted = income_stmt.applymap(lambda x: f"${x:,.0f}" if isinstance(x, (int, float)) else x)
                st.dataframe(income_stmt_formatted)
                
                # 主要項目のグラフ表示
                if 'TotalRevenue' in income_stmt.index:
                    revenue = income_stmt.loc['TotalRevenue']
                    net_income = income_stmt.loc['NetIncome'] if 'NetIncome' in income_stmt.index else pd.Series([0] * len(revenue), index=revenue.index)
                    
                    df_plot = pd.DataFrame({
                        'Revenue': revenue,
                        'Net Income': net_income
                    })
                    
                    fig = px.bar(df_plot, title=f"{st.session_state.target_company}の売上と利益")
                    # Y軸のフォーマットを設定
                    fig.update_layout(yaxis=dict(tickformat=",.0f", tickprefix="$"))
                    st.plotly_chart(fig, use_container_width=True)
        
        with fin_tabs[1]:
            if 'balance_sheet' in st.session_state.ma_tool.financial_data:
                balance_sheet = st.session_state.ma_tool.financial_data['balance_sheet']
                
                # 数値を3桁区切りでフォーマット
                balance_sheet_formatted = balance_sheet.applymap(lambda x: f"${x:,.0f}" if isinstance(x, (int, float)) else x)
                st.dataframe(balance_sheet_formatted)
                
                # 主要項目のグラフ表示
                if 'TotalAssets' in balance_sheet.index and 'TotalLiabilities' in balance_sheet.index:
                    assets = balance_sheet.loc['TotalAssets']
                    liabilities = balance_sheet.loc['TotalLiabilities']
                    equity = assets - liabilities
                    
                    df_plot = pd.DataFrame({
                        'Assets': assets,
                        'Liabilities': liabilities,
                        'Equity': equity
                    })
                    
                    fig = px.bar(df_plot, title=f"{st.session_state.target_company}の資産・負債・資本")
                    # Y軸のフォーマットを設定
                    fig.update_layout(yaxis=dict(tickformat=",.0f", tickprefix="$"))
                    st.plotly_chart(fig, use_container_width=True)
        
        with fin_tabs[2]:
            if 'cash_flow' in st.session_state.ma_tool.financial_data:
                cash_flow = st.session_state.ma_tool.financial_data['cash_flow']
                
                # 数値を3桁区切りでフォーマット
                cash_flow_formatted = cash_flow.applymap(lambda x: f"${x:,.0f}" if isinstance(x, (int, float)) else x)
                st.dataframe(cash_flow_formatted)
                
                # 主要項目のグラフ表示
                if 'OperatingCashFlow' in cash_flow.index:
                    operating_cf = cash_flow.loc['OperatingCashFlow']
                    investing_cf = cash_flow.loc['InvestingCashFlow'] if 'InvestingCashFlow' in cash_flow.index else pd.Series([0] * len(operating_cf), index=operating_cf.index)
                    financing_cf = cash_flow.loc['FinancingCashFlow'] if 'FinancingCashFlow' in cash_flow.index else pd.Series([0] * len(operating_cf), index=operating_cf.index)
                    
                    df_plot = pd.DataFrame({
                        'Operating CF': operating_cf,
                        'Investing CF': investing_cf,
                        'Financing CF': financing_cf
                    })
                    
                    fig = px.bar(df_plot, title=f"{st.session_state.target_company}のキャッシュフロー")
                    # Y軸のフォーマットを設定
                    fig.update_layout(yaxis=dict(tickformat=",.0f", tickprefix="$"))
                    st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("左のサイドバーから企業ティッカーを入力し、「企業データをロード」ボタンをクリックしてください。")

# タブ2: 異常検知
with tab2:
    if st.session_state.target_company:
        st.markdown(f"<h2 class='section-header'>{st.session_state.target_company}の財務異常検知</h2>", unsafe_allow_html=True)
        
        if st.button("異常検知分析を実行", key="run_anomaly"):
            with st.spinner("異常検知分析を実行中..."):
                success, result = st.session_state.ma_tool.detect_financial_anomalies()
                
                if success:
                    st.session_state.analysis_results['anomaly'] = result
                    st.success("異常検知分析が完了しました。")
                else:
                    st.error(f"異常検知分析に失敗しました。エラー: {result}")
        
        if 'anomaly' in st.session_state.analysis_results:
            result = st.session_state.analysis_results['anomaly']
            
            # 異常検知結果の表示
            st.markdown("<h3 class='subsection-header'>異常検知結果</h3>", unsafe_allow_html=True)
            
            # 図の表示
            st.plotly_chart(result['figure'], use_container_width=True)
            
            # 異常値のリスト
            st.markdown("<h3 class='subsection-header'>検出された異常値</h3>", unsafe_allow_html=True)
            
            if len(result['anomaly_data']) > 0:
                anomaly_df = result['anomaly_data'][['Close', 'Volume', 'Returns', 'Volatility']]
                
                # 数値フォーマットの適用
                anomaly_df_formatted = anomaly_df.copy()
                anomaly_df_formatted['Close'] = anomaly_df_formatted['Close'].apply(lambda x: f"${x:,.2f}")
                anomaly_df_formatted['Volume'] = anomaly_df_formatted['Volume'].apply(lambda x: f"{x:,.0f}")
                anomaly_df_formatted['Returns'] = anomaly_df_formatted['Returns'].apply(lambda x: f"{x:.4f}")
                anomaly_df_formatted['Volatility'] = anomaly_df_formatted['Volatility'].apply(lambda x: f"{x:.4f}")
                
                anomaly_df_formatted.index = anomaly_df_formatted.index.strftime('%Y-%m-%d')
                st.dataframe(anomaly_df_formatted)
                
                st.markdown("<div class='info-box'>", unsafe_allow_html=True)
                st.markdown(f"**検出された異常値の数:** {len(result['anomaly_dates'])}")
                st.markdown(f"**異常値の割合:** {len(result['anomaly_dates']) / len(result['data']) * 100:.2f}%")
                st.markdown("</div>", unsafe_allow_html=True)
                
                # 異常値の特徴
                st.markdown("<h3 class='subsection-header'>異常値の特徴</h3>", unsafe_allow_html=True)
                
                # 通常データと異常データの比較
                normal_data = result['data'][result['data']['Anomaly'] == 0]
                anomaly_data = result['data'][result['data']['Anomaly'] == 1]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**通常データの特徴**")
                    st.markdown(f"平均リターン: {normal_data['Returns'].mean():.4f}")
                    st.markdown(f"平均ボラティリティ: {normal_data['Volatility'].mean():.4f}")
                    st.markdown(f"平均出来高変化: {normal_data['Volume_Change'].mean():.4f}")
                
                with col2:
                    st.markdown("**異常データの特徴**")
                    st.markdown(f"平均リターン: {anomaly_data['Returns'].mean():.4f}")
                    st.markdown(f"平均ボラティリティ: {anomaly_data['Volatility'].mean():.4f}")
                    st.markdown(f"平均出来高変化: {anomaly_data['Volume_Change'].mean():.4f}")
            else:
                st.info("異常値は検出されませんでした。")
    else:
        st.info("左のサイドバーから企業ティッカーを入力し、「企業データをロード」ボタンをクリックしてください。")

# タブ3: バリュエーション
with tab3:
    if st.session_state.target_company:
        st.markdown(f"<h2 class='section-header'>{st.session_state.target_company}のバリュエーション分析</h2>", unsafe_allow_html=True)
        
        # バリュエーション設定
        col1, col2 = st.columns(2)
        
        with col1:
            dcf_enabled = st.checkbox("DCF法", value=True)
        
        with col2:
            multiples_enabled = st.checkbox("マルチプル法", value=True)
        
        methods = []
        if dcf_enabled:
            methods.append("DCF")
        if multiples_enabled:
            methods.append("Multiples")
        
        if st.button("バリュエーション分析を実行", key="run_valuation"):
            if len(methods) > 0:
                with st.spinner("バリュエーション分析を実行中..."):
                    success, result = st.session_state.ma_tool.calculate_valuation(methods=methods)
                    
                    if success:
                        st.session_state.analysis_results['valuation'] = result
                        st.success("バリュエーション分析が完了しました。")
                    else:
                        st.error(f"バリュエーション分析に失敗しました。エラー: {result}")
            else:
                st.warning("少なくとも1つのバリュエーション手法を選択してください。")
        
        if 'valuation' in st.session_state.analysis_results:
            result = st.session_state.analysis_results['valuation']
            
            # バリュエーション結果の表示
            st.markdown("<h3 class='subsection-header'>バリュエーション結果</h3>", unsafe_allow_html=True)
            
            # 図の表示
            if 'figure' in result:
                st.plotly_chart(result['figure'], use_container_width=True)
            
            if 'figure_upside' in result:
                st.plotly_chart(result['figure_upside'], use_container_width=True)
            
            # バリュエーション詳細
            st.markdown("<h3 class='subsection-header'>バリュエーション詳細</h3>", unsafe_allow_html=True)
            
            valuation_tabs = st.tabs([m for m in methods] + ["比較サマリー"])
            
            # DCF法の詳細
            if "DCF" in methods and 'DCF' in result['results']:
                with valuation_tabs[methods.index("DCF")]:
                    dcf_result = result['results']['DCF']
                    
                    if 'error' in dcf_result:
                        st.error(f"DCF分析エラー: {dcf_result['error']}")
                    else:
                        st.markdown("<div class='info-box'>", unsafe_allow_html=True)
                        st.markdown(f"**企業価値 (EV):** ${dcf_result['enterprise_value']:,.0f}")
                        st.markdown(f"**株主価値:** ${dcf_result['equity_value']:,.0f}")
                        st.markdown(f"**1株あたり価値:** ${dcf_result['per_share_value']:.2f}")
                        st.markdown(f"**現在株価:** ${dcf_result['current_price']:.2f}")
                        st.markdown(f"**アップサイドポテンシャル:** {dcf_result['upside_potential']:.2f}%")
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                        # DCFパラメータ
                        st.markdown("<h4>DCF計算パラメータ</h4>", unsafe_allow_html=True)
                        params = dcf_result['parameters']
                        
                        param_col1, param_col2 = st.columns(2)
                        
                        with param_col1:
                            st.markdown(f"成長率: {params['growth_rate'] * 100:.2f}%")
                            st.markdown(f"割引率 (WACC): {params['discount_rate'] * 100:.2f}%")
                        
                        with param_col2:
                            st.markdown(f"永続成長率: {params['terminal_growth_rate'] * 100:.2f}%")
                            st.markdown(f"予測期間: {params['projection_years']}年")
            
            # マルチプル法の詳細
            if "Multiples" in methods and 'Multiples' in result['results']:
                with valuation_tabs[methods.index("Multiples")]:
                    multiple_result = result['results']['Multiples']
                    
                    if 'error' in multiple_result:
                        st.error(f"マルチプル分析エラー: {multiple_result['error']}")
                    else:
                        # EV/EBITDA
                        if 'EV_EBITDA' in multiple_result:
                            st.markdown("<h4>EV/EBITDA倍率法</h4>", unsafe_allow_html=True)
                            ev_ebitda = multiple_result['EV_EBITDA']
                            
                            st.markdown("<div class='info-box'>", unsafe_allow_html=True)
                            st.markdown(f"**採用倍率:** {ev_ebitda['multiple']:.2f}x")
                            st.markdown(f"**企業価値 (EV):** ${ev_ebitda['enterprise_value']:,.0f}")
                            st.markdown(f"**株主価値:** ${ev_ebitda['equity_value']:,.0f}")
                            st.markdown(f"**1株あたり価値:** ${ev_ebitda['per_share_value']:.2f}")
                            st.markdown(f"**アップサイドポテンシャル:** {ev_ebitda['upside_potential']:.2f}%")
                            st.markdown("</div>", unsafe_allow_html=True)
                        
                        # P/E
                        if 'PE' in multiple_result:
                            st.markdown("<h4>P/E倍率法</h4>", unsafe_allow_html=True)
                            pe = multiple_result['PE']
                            
                            st.markdown("<div class='info-box'>", unsafe_allow_html=True)
                            st.markdown(f"**採用倍率:** {pe['multiple']:.2f}x")
                            st.markdown(f"**1株あたり価値:** ${pe['per_share_value']:.2f}")
                            st.markdown(f"**アップサイドポテンシャル:** {pe['upside_potential']:.2f}%")
                            st.markdown("</div>", unsafe_allow_html=True)
            
            # 比較サマリー
            with valuation_tabs[-1]:
                if 'summary' in result:
                    st.dataframe(result['summary'])
                    
                    # 公正価値の推定
                    if len(result['summary']) > 1:
                        fair_values = result['summary'][result['summary']['Method'] != 'Current Market Price']['Value'].values
                        avg_fair_value = np.mean(fair_values)
                        min_fair_value = np.min(fair_values)
                        max_fair_value = np.max(fair_values)
                        current_price = result['summary'][result['summary']['Method'] == 'Current Market Price']['Value'].values[0]
                        
                        st.markdown("<h4>公正価値の推定レンジ</h4>", unsafe_allow_html=True)
                        
                        st.markdown("<div class='success-box'>", unsafe_allow_html=True)
                        st.markdown(f"**最低価値:** ${min_fair_value:.2f} (現在比: {(min_fair_value / current_price - 1) * 100:.2f}%)")
                        st.markdown(f"**平均価値:** ${avg_fair_value:.2f} (現在比: {(avg_fair_value / current_price - 1) * 100:.2f}%)")
                        st.markdown(f"**最高価値:** ${max_fair_value:.2f} (現在比: {(max_fair_value / current_price - 1) * 100:.2f}%)")
                        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("左のサイドバーから企業ティッカーを入力し、「企業データをロード」ボタンをクリックしてください。")

# タブ4: 類似企業
with tab4:
    if st.session_state.target_company:
        st.markdown(f"<h2 class='section-header'>{st.session_state.target_company}の類似企業分析</h2>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            market_cap_range = st.slider("時価総額許容範囲 (±%)", 10, 100, 50) / 100
        
        with col2:
            num_companies = st.slider("表示する類似企業数", 3, 10, 5)
        
        if st.button("類似企業を検索", key="run_peer_analysis"):
            with st.spinner("類似企業を検索中..."):
                success, result = st.session_state.ma_tool.find_similar_companies(
                    market_cap_range=market_cap_range, 
                    num_companies=num_companies
                )
                
                if success:
                    st.session_state.analysis_results['peers'] = result
                    st.success("類似企業の検索が完了しました。")
                else:
                    st.error(f"類似企業の検索に失敗しました。エラー: {result}")
        
        if 'peers' in st.session_state.analysis_results:
            result = st.session_state.analysis_results['peers']
            
            # 類似企業の表示
            st.markdown("<h3 class='subsection-header'>類似企業リスト</h3>", unsafe_allow_html=True)
            
            # 図の表示
            if 'figure' in result:
                st.plotly_chart(result['figure'], use_container_width=True)
            
            # 類似企業テーブル
            if 'similar_companies' in result and not result['similar_companies'].empty:
                # データの整形
                df = result['similar_companies'].copy()
                df['market_cap'] = df['market_cap'].apply(lambda x: f"${x:,.0f}")
                df['PE_ratio'] = df['PE_ratio'].apply(lambda x: f"{x:.2f}x" if pd.notnull(x) else "N/A")
                df['EV_EBITDA'] = df['EV_EBITDA'].apply(lambda x: f"{x:.2f}x" if pd.notnull(x) else "N/A")
                
                st.dataframe(df)
                
                # 類似企業の詳細比較
                st.markdown("<h3 class='subsection-header'>類似企業の詳細比較</h3>", unsafe_allow_html=True)

                # 選択した企業の詳細を表示
                selected_peer = st.selectbox("企業を選択", df['ticker'].tolist())
                
                if selected_peer:
                    with st.spinner(f"{selected_peer}の詳細情報を取得中..."):
                        peer_company = yf.Ticker(selected_peer)
                        peer_info = peer_company.info
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown(f"**会社名:** {peer_info.get('shortName', 'N/A')}")
                            st.markdown(f"**ティッカー:** {selected_peer}")
                            st.markdown(f"**業種:** {peer_info.get('industry', 'N/A')}")
                            st.markdown(f"**時価総額:** ${peer_info.get('marketCap', 0):,.0f}")
                        
                        with col2:
                            st.markdown(f"**現在株価:** ${peer_info.get('currentPrice', 0):,.2f}")
                            st.markdown(f"**P/E倍率:** {peer_info.get('trailingPE', 'N/A')}")
                            st.markdown(f"**EV/EBITDA:** {peer_info.get('enterpriseToEbitda', 'N/A')}")
                            
                            # 配当利回りの表示を修正
                            trailing_yield = peer_info.get('trailingAnnualDividendYield', 0)
                            dividend_yield = peer_info.get('dividendYield', 0)
                            
                            # trailingAnnualDividendYieldを優先的に使用
                            if trailing_yield > 0:
                                st.markdown(f"**配当利回り:** {trailing_yield * 100:.2f}%")
                            else:
                                # dividendYieldの値が1未満かどうかをチェック
                                if dividend_yield < 1:
                                    # 小数形式なので、100を掛けてパーセンテージ表示にする
                                    st.markdown(f"**配当利回り:** {dividend_yield * 100:.2f}%")
                                else:
                                    # すでにパーセンテージ形式の場合はそのまま表示
                                    st.markdown(f"**配当利回り:** {dividend_yield:.2f}%")
                        
                        # 企業概要
                        st.markdown("<h4>ビジネス概要</h4>", unsafe_allow_html=True)
                        st.markdown(f"{peer_info.get('longBusinessSummary', 'N/A')}")
                        
                        # 株価比較
                        st.markdown("<h4>株価パフォーマンス比較</h4>", unsafe_allow_html=True)
                        
                        try:
                            # 株価データの取得（1年分）
                            end_date = datetime.now()
                            start_date = end_date - timedelta(days=365)
                            
                            target_data = yf.download(st.session_state.target_company, start=start_date, end=end_date)
                            peer_data = yf.download(selected_peer, start=start_date, end=end_date)
                            
                            # 相対パフォーマンスの計算
                            target_perf = target_data['Close'] / target_data['Close'].iloc[0] * 100
                            peer_perf = peer_data['Close'] / peer_data['Close'].iloc[0] * 100
                            
                            # データの結合
                            perf_df = pd.DataFrame({
                                st.session_state.target_company: target_perf,
                                selected_peer: peer_perf
                            })
                            
                            # パフォーマンス比較グラフ
                            fig = px.line(perf_df, title=f"相対株価パフォーマンス比較 (過去1年)")
                            fig.update_layout(yaxis_title='相対パフォーマンス (%)', xaxis_title='日付')
                            st.plotly_chart(fig, use_container_width=True)
                        except:
                            st.warning("株価パフォーマンス比較データの取得に失敗しました。")
                        
                        # 財務指標比較
                        st.markdown("<h4>財務指標比較</h4>", unsafe_allow_html=True)
                        
                        try:
                            # 対象企業の財務データ
                            target_info = st.session_state.ma_tool.financial_data['info']
                            
                            # 配当利回りの計算（trailingAnnualDividendYieldを優先）
                            target_trailing_yield = target_info.get('trailingAnnualDividendYield', 0)
                            target_dividend_yield = target_info.get('dividendYield', 0)
                            
                            if target_trailing_yield > 0:
                                target_yield_display = target_trailing_yield * 100
                            else:
                                if target_dividend_yield < 1:
                                    target_yield_display = target_dividend_yield * 100
                                else:
                                    target_yield_display = target_dividend_yield
                            
                            # 類似企業の配当利回りの計算
                            peer_trailing_yield = peer_info.get('trailingAnnualDividendYield', 0)
                            peer_dividend_yield = peer_info.get('dividendYield', 0)
                            
                            if peer_trailing_yield > 0:
                                peer_yield_display = peer_trailing_yield * 100
                            else:
                                if peer_dividend_yield < 1:
                                    peer_yield_display = peer_dividend_yield * 100
                                else:
                                    peer_yield_display = peer_dividend_yield
                            
                            # 比較データの作成
                            comparison_data = {
                                '指標': [
                                    '時価総額', 'P/E倍率', 'EV/EBITDA', '配当利回り', 
                                    '営業利益率', '純利益率', 'ROE', 'ROA'
                                ],
                                st.session_state.target_company: [
                                    target_info.get('marketCap', 0),
                                    target_info.get('trailingPE', 0),
                                    target_info.get('enterpriseToEbitda', 0),
                                    target_yield_display,
                                    target_info.get('operatingMargins', 0) * 100,
                                    target_info.get('profitMargins', 0) * 100,
                                    target_info.get('returnOnEquity', 0) * 100,
                                    target_info.get('returnOnAssets', 0) * 100
                                ],
                                selected_peer: [
                                    peer_info.get('marketCap', 0),
                                    peer_info.get('trailingPE', 0),
                                    peer_info.get('enterpriseToEbitda', 0),
                                    peer_yield_display,
                                    peer_info.get('operatingMargins', 0) * 100,
                                    peer_info.get('profitMargins', 0) * 100,
                                    peer_info.get('returnOnEquity', 0) * 100,
                                    peer_info.get('returnOnAssets', 0) * 100
                                ]
                            }
                            
                            # データフレームに変換
                            comparison_df = pd.DataFrame(comparison_data)
                            comparison_df = comparison_df.set_index('指標')
                            
                            # 表形式で表示
                            st.dataframe(comparison_df)
                            
                            # レーダーチャートでの比較
                            categories = [
                                'P/E倍率', 'EV/EBITDA', '配当利回り', 
                                '営業利益率', '純利益率', 'ROE', 'ROA'
                            ]
                            
                            fig = go.Figure()
                            
                            # 対象企業のデータ
                            target_values = [
                                target_info.get('trailingPE', 0),
                                target_info.get('enterpriseToEbitda', 0),
                                target_yield_display,
                                target_info.get('operatingMargins', 0) * 100,
                                target_info.get('profitMargins', 0) * 100,
                                target_info.get('returnOnEquity', 0) * 100,
                                target_info.get('returnOnAssets', 0) * 100
                            ]
                            
                            # P/EとEV/EBITDAは低いほど良いので反転
                            normalized_target = [
                                1 / max(target_values[0], 0.01) * 100,
                                1 / max(target_values[1], 0.01) * 100,
                                target_values[2], target_values[3], target_values[4], target_values[5], target_values[6]
                            ]
                            
                            # 類似企業のデータ
                            peer_values = [
                                peer_info.get('trailingPE', 0),
                                peer_info.get('enterpriseToEbitda', 0),
                                peer_yield_display,
                                peer_info.get('operatingMargins', 0) * 100,
                                peer_info.get('profitMargins', 0) * 100,
                                peer_info.get('returnOnEquity', 0) * 100,
                                peer_info.get('returnOnAssets', 0) * 100
                            ]
                            
                            # P/EとEV/EBITDAは低いほど良いので反転
                            normalized_peer = [
                                1 / max(peer_values[0], 0.01) * 100,
                                1 / max(peer_values[1], 0.01) * 100,
                                peer_values[2], peer_values[3], peer_values[4], peer_values[5], peer_values[6]
                            ]
                            
                            # レーダーチャート作成
                            fig.add_trace(go.Scatterpolar(
                                r=normalized_target,
                                theta=categories,
                                fill='toself',
                                name=st.session_state.target_company
                            ))
                            
                            fig.add_trace(go.Scatterpolar(
                                r=normalized_peer,
                                theta=categories,
                                fill='toself',
                                name=selected_peer
                            ))
                            
                            fig.update_layout(
                                polar=dict(
                                    radialaxis=dict(
                                        visible=True,
                                        range=[0, 100]
                                    )
                                ),
                                showlegend=True,
                                title="企業間財務指標比較"
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.warning(f"財務指標比較データの取得に失敗しました。エラー: {e}")
            else:
                st.info("類似企業が見つかりませんでした。")
    else:
        st.info("左のサイドバーから企業ティッカーを入力し、「企業データをロード」ボタンをクリックしてください。")

# タブ5: M&Aシミュレーション
with tab5:
    if st.session_state.target_company:
        st.markdown(f"<h2 class='section-header'>M&Aシミュレーション ({st.session_state.target_company} as Acquirer)</h2>", unsafe_allow_html=True)
        
        target_ticker = st.text_input("買収対象企業のティッカーを入力", "MSFT", help="例: MSFT (Microsoft Corp.), GOOGL (Alphabet Inc.)")
        
        # シナジータイプの選択
        synergy_types = []
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.checkbox("収益シナジー", value=True):
                synergy_types.append("Revenue")
        
        with col2:
            if st.checkbox("コストシナジー", value=True):
                synergy_types.append("Cost")
        
        with col3:
            if st.checkbox("財務シナジー", value=True):
                synergy_types.append("Financial")
        
        if st.button("M&Aシミュレーションを実行", key="run_ma_simulation"):
            if target_ticker and target_ticker != st.session_state.target_company:
                with st.spinner(f"{st.session_state.target_company}による{target_ticker}買収のシミュレーションを実行中..."):
                    success, result = st.session_state.ma_tool.estimate_synergy(
                        target_ticker=target_ticker,
                        synergy_types=synergy_types
                    )
                    
                    if success:
                        st.session_state.analysis_results['synergy'] = result
                        st.success("M&Aシミュレーションが完了しました。")
                    else:
                        st.error(f"M&Aシミュレーションに失敗しました。エラー: {result}")
            else:
                st.warning("有効な買収対象企業のティッカーを入力してください。")
        
        if 'synergy' in st.session_state.analysis_results:
            result = st.session_state.analysis_results['synergy']
            
            # シナジー結果の表示
            st.markdown("<h3 class='subsection-header'>シナジー効果推定</h3>", unsafe_allow_html=True)
            
            # 図の表示
            if 'figures' in result and len(result['figures']) > 0:
                for fig in result['figures']:
                    st.plotly_chart(fig, use_container_width=True)
            
            # シナジー詳細
            if 'results' in result:
                synergy_results = result['results']
                
                st.markdown("<h3 class='subsection-header'>シナジー効果の詳細</h3>", unsafe_allow_html=True)
                
                synergy_tabs = st.tabs([s.capitalize() for s in synergy_types] + ["総合評価"])
                
                # 収益シナジー
                if "Revenue" in synergy_types and 'Revenue' in synergy_results:
                    with synergy_tabs[synergy_types.index("Revenue")]:
                        revenue_synergy = synergy_results['Revenue']
                        
                        st.markdown("<div class='info-box'>", unsafe_allow_html=True)
                        st.markdown(f"**買収企業売上:** ${revenue_synergy['acquirer_revenue']:,.0f}")
                        st.markdown(f"**対象企業売上:** ${revenue_synergy['target_revenue']:,.0f}")
                        st.markdown(f"**合計売上:** ${revenue_synergy['combined_revenue']:,.0f}")
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                        st.markdown("<h4>推定収益シナジー</h4>", unsafe_allow_html=True)
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.markdown("<div class='info-box'>", unsafe_allow_html=True)
                            st.markdown("**保守的推定**")
                            st.markdown(f"${revenue_synergy['synergy_low']:,.0f}")
                            # ゼロ除算エラーを防ぐための条件分岐
                            if 'combined_revenue' in revenue_synergy and revenue_synergy['combined_revenue'] > 0:
                                st.markdown(f"(合計売上の{revenue_synergy['synergy_low'] / revenue_synergy['combined_revenue'] * 100:.1f}%)")
                            else:
                                st.markdown("(合計売上の計算不可)")
                            st.markdown("</div>", unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown("<div class='info-box'>", unsafe_allow_html=True)
                            st.markdown("**標準推定**")
                            st.markdown(f"${revenue_synergy['synergy_mid']:,.0f}")
                            # ゼロ除算エラーを防ぐための条件分岐
                            if 'combined_revenue' in revenue_synergy and revenue_synergy['combined_revenue'] > 0:
                                st.markdown(f"(合計売上の{revenue_synergy['synergy_mid'] / revenue_synergy['combined_revenue'] * 100:.1f}%)")
                            else:
                                st.markdown("(合計売上の計算不可)")
                            st.markdown("</div>", unsafe_allow_html=True)
                        
                        with col3:
                            st.markdown("<div class='info-box'>", unsafe_allow_html=True)
                            st.markdown("**積極的推定**")
                            st.markdown(f"${revenue_synergy['synergy_high']:,.0f}")
                            # ゼロ除算エラーを防ぐための条件分岐
                            if 'combined_revenue' in revenue_synergy and revenue_synergy['combined_revenue'] > 0:
                                st.markdown(f"(合計売上の{revenue_synergy['synergy_high'] / revenue_synergy['combined_revenue'] * 100:.1f}%)")
                            else:
                                st.markdown("(合計売上の計算不可)")
                            st.markdown("</div>", unsafe_allow_html=True)
                
                # コストシナジー
                if "Cost" in synergy_types and 'Cost' in synergy_results:
                    with synergy_tabs[synergy_types.index("Cost")]:
                        cost_synergy = synergy_results['Cost']
                        
                        st.markdown("<div class='info-box'>", unsafe_allow_html=True)
                        st.markdown(f"**買収企業コスト:** ${cost_synergy['acquirer_opex']:,.0f}")
                        st.markdown(f"**対象企業コスト:** ${cost_synergy['target_opex']:,.0f}")
                        st.markdown(f"**合計コスト:** ${cost_synergy['combined_opex']:,.0f}")
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                        st.markdown("<h4>推定コストシナジー</h4>", unsafe_allow_html=True)
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.markdown("<div class='info-box'>", unsafe_allow_html=True)
                            st.markdown("**保守的推定**")
                            st.markdown(f"${cost_synergy['synergy_low']:,.0f}")
                            # ゼロ除算エラーを防ぐための条件分岐
                            if 'combined_opex' in cost_synergy and cost_synergy['combined_opex'] > 0:
                                st.markdown(f"(合計コストの{cost_synergy['synergy_low'] / cost_synergy['combined_opex'] * 100:.1f}%)")
                            else:
                                st.markdown("(合計コストの計算不可)")
                            st.markdown("</div>", unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown("<div class='info-box'>", unsafe_allow_html=True)
                            st.markdown("**標準推定**")
                            st.markdown(f"${cost_synergy['synergy_mid']:,.0f}")
                            # ゼロ除算エラーを防ぐための条件分岐
                            if 'combined_opex' in cost_synergy and cost_synergy['combined_opex'] > 0:
                                st.markdown(f"(合計コストの{cost_synergy['synergy_mid'] / cost_synergy['combined_opex'] * 100:.1f}%)")
                            else:
                                st.markdown("(合計コストの計算不可)")
                            st.markdown("</div>", unsafe_allow_html=True)
                        
                        with col3:
                            st.markdown("<div class='info-box'>", unsafe_allow_html=True)
                            st.markdown("**積極的推定**")
                            st.markdown(f"${cost_synergy['synergy_high']:,.0f}")
                            # ゼロ除算エラーを防ぐための条件分岐
                            if 'combined_opex' in cost_synergy and cost_synergy['combined_opex'] > 0:
                                st.markdown(f"(合計コストの{cost_synergy['synergy_high'] / cost_synergy['combined_opex'] * 100:.1f}%)")
                            else:
                                st.markdown("(合計コストの計算不可)")
                            st.markdown("</div>", unsafe_allow_html=True)
                
                # 財務シナジー
                if "Financial" in synergy_types and 'Financial' in synergy_results:
                    with synergy_tabs[synergy_types.index("Financial")]:
                        financial_synergy = synergy_results['Financial']
                        
                        st.markdown("<h4>財務シナジー</h4>", unsafe_allow_html=True)
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.markdown("<div class='info-box'>", unsafe_allow_html=True)
                            st.markdown("**保守的推定**")
                            st.markdown(f"統合後WACC: {financial_synergy['combined_wacc_low'] * 100:.2f}%")
                            st.markdown(f"シナジー価値: ${financial_synergy['synergy_low']:,.0f}")
                            st.markdown("</div>", unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown("<div class='info-box'>", unsafe_allow_html=True)
                            st.markdown("**標準推定**")
                            st.markdown(f"統合後WACC: {financial_synergy['combined_wacc_mid'] * 100:.2f}%")
                            st.markdown(f"シナジー価値: ${financial_synergy['synergy_mid']:,.0f}")
                            st.markdown("</div>", unsafe_allow_html=True)
                        
                        with col3:
                            st.markdown("<div class='info-box'>", unsafe_allow_html=True)
                            st.markdown("**積極的推定**")
                            st.markdown(f"統合後WACC: {financial_synergy['combined_wacc_high'] * 100:.2f}%")
                            st.markdown(f"シナジー価値: ${financial_synergy['synergy_high']:,.0f}")
                            st.markdown("</div>", unsafe_allow_html=True)
                
                # 総合評価
                with synergy_tabs[-1]:
                    if 'Total' in synergy_results:
                        total_synergy = synergy_results['Total']
                        
                        st.markdown("<h4>総合シナジー効果</h4>", unsafe_allow_html=True)
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.markdown("<div class='info-box'>", unsafe_allow_html=True)
                            st.markdown("**保守的推定**")
                            st.markdown(f"${total_synergy['synergy_low']:,.0f}")
                            # ゼロ除算エラーを防ぐための条件分岐
                            if 'combined_revenue' in total_synergy and total_synergy['combined_revenue'] > 0:
                                st.markdown(f"(合計売上の{total_synergy['synergy_low'] / total_synergy['combined_revenue'] * 100:.1f}%)")
                            else:
                                st.markdown("(合計売上の計算不可)")
                            st.markdown("</div>", unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown("<div class='info-box'>", unsafe_allow_html=True)
                            st.markdown("**標準推定**")
                            st.markdown(f"${total_synergy['synergy_mid']:,.0f}")
                            # ゼロ除算エラーを防ぐための条件分岐
                            if 'combined_revenue' in total_synergy and total_synergy['combined_revenue'] > 0:
                                st.markdown(f"(合計売上の{total_synergy['synergy_mid'] / total_synergy['combined_revenue'] * 100:.1f}%)")
                            else:
                                st.markdown("(合計売上の計算不可)")
                            st.markdown("</div>", unsafe_allow_html=True)
                        
                        with col3:
                            st.markdown("<div class='info-box'>", unsafe_allow_html=True)
                            st.markdown("**積極的推定**")
                            st.markdown(f"${total_synergy['synergy_high']:,.0f}")
                            # ゼロ除算エラーを防ぐための条件分岐
                            if 'combined_revenue' in total_synergy and total_synergy['combined_revenue'] > 0:
                                st.markdown(f"(合計売上の{total_synergy['synergy_high'] / total_synergy['combined_revenue'] * 100:.1f}%)")
                            else:
                                st.markdown("(合計売上の計算不可)")
                            st.markdown("</div>", unsafe_allow_html=True)
                        
                        # 買収判断
                        if 'acquisition_data' in result:
                            st.markdown("<h4>買収判断</h4>", unsafe_allow_html=True)
                            
                            acq_data = result['acquisition_data']
                            target_market_cap = acq_data[acq_data['Category'] == 'Target Market Cap']['Value'].values[0]
                            target_with_premium = acq_data[acq_data['Category'] == 'Target with Premium']['Value'].values[0]
                            
                            # 中位シナジー価値
                            synergy_mid = total_synergy['synergy_mid']
                            
                            # 買収価値
                            acquisition_value = target_with_premium - synergy_mid
                            
                            st.markdown("<div class='success-box'>", unsafe_allow_html=True)
                            st.markdown(f"**対象企業時価総額:** ${target_market_cap:,.0f}")
                            st.markdown(f"**プレミアム付き買収コスト:** ${target_with_premium:,.0f}")
                            st.markdown(f"**シナジー価値 (中位推定):** ${synergy_mid:,.0f}")
                            st.markdown(f"**実質買収コスト:** ${acquisition_value:,.0f}")
                            
                            # 買収判断
                            if synergy_mid > target_with_premium * 0.3:  # シナジーがプレミアムを上回る
                                st.markdown("**判断:** ✅ 買収を推奨 - シナジー効果がプレミアムを上回ります")
                            elif synergy_mid > target_with_premium * 0.1:  # シナジーがプレミアムの一部を補填
                                st.markdown("**判断:** ⚠️ 条件付き推奨 - シナジー効果が一部プレミアムを相殺します")
                            else:  # シナジーが少ない
                                st.markdown("**判断:** ❌ 買収を推奨しない - シナジー効果が限定的です")
                            
                            st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("左のサイドバーから企業ティッカーを入力し、「企業データをロード」ボタンをクリックしてください。")

# レポート出力機能
st.sidebar.markdown("---")
st.sidebar.markdown("## レポート出力")

if st.session_state.target_company and st.sidebar.button("分析レポートを生成"):
    with st.spinner("レポートを生成中..."):
        # ExcelファイルをPandas ExcelWriterで作成
        output = BytesIO()
        writer = pd.ExcelWriter(output, engine='xlsxwriter')
        
        # 1. 企業概要シート
        if 'info' in st.session_state.ma_tool.financial_data:
            info = st.session_state.ma_tool.financial_data['info']
            info_df = pd.DataFrame([
                {'項目': '会社名', '値': info.get('shortName', 'N/A')},
                {'項目': 'ティッカー', '値': st.session_state.target_company},
                {'項目': '業種', '値': info.get('industry', 'N/A')},
                {'項目': 'セクター', '値': info.get('sector', 'N/A')},
                {'項目': '時価総額', '値': info.get('marketCap', 0)},
                {'項目': '現在株価', '値': info.get('currentPrice', 0)},
                {'項目': 'P/E倍率', '値': info.get('trailingPE', 'N/A')},
                {'項目': 'EPS', '値': info.get('trailingEps', 0)},
                {'項目': '配当利回り', '値': info.get('dividendYield', 0) * 100},
                {'項目': '営業利益率', '値': info.get('operatingMargins', 0) * 100},
                {'項目': '純利益率', '値': info.get('profitMargins', 0) * 100},
                {'項目': 'ROE', '値': info.get('returnOnEquity', 0) * 100},
                {'項目': 'ROA', '値': info.get('returnOnAssets', 0) * 100}
            ])
            info_df.to_excel(writer, sheet_name='企業概要', index=False)
        
        # 2. 財務諸表
        if 'income_statement' in st.session_state.ma_tool.financial_data:
            st.session_state.ma_tool.financial_data['income_statement'].to_excel(writer, sheet_name='損益計算書')
        
        if 'balance_sheet' in st.session_state.ma_tool.financial_data:
            st.session_state.ma_tool.financial_data['balance_sheet'].to_excel(writer, sheet_name='貸借対照表')
        
        if 'cash_flow' in st.session_state.ma_tool.financial_data:
            st.session_state.ma_tool.financial_data['cash_flow'].to_excel(writer, sheet_name='キャッシュフロー')
        
        # 3. 異常検知結果
        if 'anomaly' in st.session_state.analysis_results:
            anomaly_result = st.session_state.analysis_results['anomaly']
            if 'anomaly_data' in anomaly_result:
                anomaly_result['anomaly_data'].to_excel(writer, sheet_name='異常検知')
        
        # 4. バリュエーション結果
        if 'valuation' in st.session_state.analysis_results:
            valuation_result = st.session_state.analysis_results['valuation']
            if 'summary' in valuation_result:
                valuation_result['summary'].to_excel(writer, sheet_name='バリュエーション', index=False)
        
        # 5. 類似企業
        if 'peers' in st.session_state.analysis_results:
            peers_result = st.session_state.analysis_results['peers']
            if 'similar_companies' in peers_result:
                peers_result['similar_companies'].to_excel(writer, sheet_name='類似企業', index=False)
        
        # 6. シナジー分析
        if 'synergy' in st.session_state.analysis_results:
            synergy_result = st.session_state.analysis_results['synergy']
            if 'synergy_data' in synergy_result:
                pd.DataFrame(synergy_result['synergy_data']).to_excel(writer, sheet_name='シナジー分析', index=False)
        
        # ExcelWriterを保存して出力
        writer.save()
        output.seek(0)
        
        # ダウンロードリンクの生成
        b64 = base64.b64encode(output.read()).decode()
        href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{st.session_state.target_company}_分析レポート.xlsx">Excelレポートをダウンロード</a>'
        st.sidebar.markdown(href, unsafe_allow_html=True)
        
        st.sidebar.success("レポートが生成されました。上のリンクからダウンロードしてください。")

# フッター
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: gray; font-size: 0.8em;">
        © 2025 M&A Analytics Platform | Science & Technology Group Demo
    </div>
    """,
    unsafe_allow_html=True
)