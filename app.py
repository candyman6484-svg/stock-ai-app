import streamlit as st
import google.generativeai as genai
import FinanceDataReader as fdr
import yfinance as yf
import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import re
import json

# -----------------------------------------------------------
# [설정] 페이지 기본 설정
# -----------------------------------------------------------
st.set_page_config(
    page_title="AI 주식 비서 (가치+차트+수급+옵션)",
    page_icon="🦅",
    layout="centered"
)

# -----------------------------------------------------------
# [설정] API 키 
# -----------------------------------------------------------
try:
    if "GOOGLE_API_KEY" in st.secrets:
        GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    else:
        GOOGLE_API_KEY = "여기에_키를_넣으세요"

    clean_key = re.sub(r'[^a-zA-Z0-9_\-]', '', GOOGLE_API_KEY)
    genai.configure(api_key=clean_key)
    model = genai.GenerativeModel('gemini-3-flash-preview')
except Exception as e:
    st.error(f"API 키 설정 오류: {e}")

HEADERS = {
    'User-Agent': 'Mozilla/5.0',
    'Accept-Language': 'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7'
}

# --- 기술적 지표 계산 함수 (기존 유지) ---
def add_technical_indicators(df):
    if len(df) < 20: return {}
    info = {}
    
    if len(df) >= 365:
        ma365 = df['Close'].rolling(window=365).mean().iloc[-1]
        info['365일_이동평균선'] = int(ma365)
        info['365일선_위치'] = "주가가 365일선 위에 있음 (장기상승세)" if df['Close'].iloc[-1] > ma365 else "주가가 365일선 아래에 있음 (장기하락세/저평가)"
    else:
        info['365일_이동평균선'] = "데이터 부족"

    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['Std'] = df['Close'].rolling(window=20).std()
    df['Upper'] = df['MA20'] + (df['Std'] * 2)
    df['Lower'] = df['MA20'] - (df['Std'] * 2)
    
    current_price = df['Close'].iloc[-1]
    info['볼린저밴드_상단'] = int(df['Upper'].iloc[-1])
    info['볼린저밴드_하단'] = int(df['Lower'].iloc[-1])
    
    if current_price >= info['볼린저밴드_상단']:
        info['볼린저밴드_상태'] = "과매수 구간 (단기 고점)"
    elif current_price <= info['볼린저밴드_하단']:
        info['볼린저밴드_상태'] = "과매도 구간 (단기 저점/반등기대)"
    else:
        info['볼린저밴드_상태'] = "밴드 내 등락 중"

    one_year_df = df[-250:] 
    bins = np.linspace(one_year_df['Low'].min(), one_year_df['High'].max(), 20)
    one_year_df['PriceBin'] = pd.cut(one_year_df['Close'], bins)
    max_vol_bin = one_year_df.groupby('PriceBin')['Volume'].sum().idxmax()
    
    info['최대_매물대_가격구간'] = str(max_vol_bin)
    if current_price < max_vol_bin.mid * 0.97:
        info['매물대_분석'] = "주가 위에 두터운 매물벽 존재 (저항)"
    elif current_price > max_vol_bin.mid * 1.03:
        info['매물대_분석'] = "주가가 매물벽을 뚫고 지지받는 중"
    else:
        info['매물대_분석'] = "최대 매물대 구간에서 힘겨루기 중"
    return info

# --- ★ [새로 추가됨] 옵션 데이터 계산 함수 ★ ---
def get_options_data(stock):
    try:
        options = stock.options
        if not options: return "옵션 데이터 없음"
        
        # 가장 가까운 만기일(보통 이번 주 금요일) 선택
        nearest_expiry = options[0]
        chain = stock.option_chain(nearest_expiry)
        
        calls = chain.calls
        puts = chain.puts
        
        if calls.empty and puts.empty: return "옵션 체인 데이터 없음"
        
        # 1. 미결제약정(Open Interest) 및 Put/Call Ratio
        call_oi = calls['openInterest'].sum()
        put_oi = puts['openInterest'].sum()
        total_oi = call_oi + put_oi
        pc_ratio = round(put_oi / call_oi, 2) if call_oi > 0 else 0
        
        # 2. Max Pain (마켓 메이커들이 가장 이득을 보는/옵션 매수자가 가장 큰 손실을 보는 가격)
        strikes = set(calls['strike']).union(set(puts['strike']))
        max_pain = 0
        min_loss = float('inf')
        
        for strike in strikes:
            call_loss = calls[calls['strike'] < strike]['openInterest'] * (strike - calls[calls['strike'] < strike]['strike'])
            put_loss = puts[puts['strike'] > strike]['openInterest'] * (puts[puts['strike'] > strike]['strike'] - strike)
            total_loss = call_loss.sum() + put_loss.sum()
            
            if total_loss < min_loss:
                min_loss = total_loss
                max_pain = strike

        # 3. 내재 변동성(Implied Volatility) - 등가격(ATM) 근처 평균
        current_price = stock.history(period="1d")['Close'].iloc[-1]
        atm_call = calls.iloc[(calls['strike'] - current_price).abs().argsort()[:1]]
        atm_put = puts.iloc[(puts['strike'] - current_price).abs().argsort()[:1]]
        
        atm_iv = 0
        if not atm_call.empty and not atm_put.empty:
            atm_iv = (atm_call['impliedVolatility'].values[0] + atm_put['impliedVolatility'].values[0]) / 2
            atm_iv = round(atm_iv * 100, 2)

        return {
            "분석기준_만기일": nearest_expiry,
            "풋콜비율(PCR)": pc_ratio,
            "총_미결제약정(OI)": int(total_oi),
            "맥스페인(Max_Pain)": max_pain,
            "등가격_내재변동성(IV)": f"{atm_iv}%"
        }
    except Exception as e:
        return f"옵션 데이터 수집 실패: {str(e)}"

# --- 데이터 수집 함수 (기존 유지) ---
def get_kr_stock_code(name):
    try:
        df = fdr.StockListing('KRX')
        matches = df[df['Name'] == name]['Code'].values
        return matches[0] if len(matches) > 0 else None
    except: return None

def get_naver_data(code):
    data = {"시장": "Korea"}
    try:
        df = fdr.DataReader(code)
        if not df.empty:
            data["주가"] = f"{int(df.iloc[-1]['Close']):,}원"
            data["기술적_지표"] = add_technical_indicators(df)
    except: pass

    try:
        url = f'https://finance.naver.com/item/main.naver?code={code}'
        res = requests.get(url, headers=HEADERS)
        res.encoding = 'EUC-KR'
        dfs = pd.read_html(res.text, match='매출액')
        if dfs:
            df = dfs[0].set_index(dfs[0].columns[0])
            data["재무"] = df.iloc[:, :4].to_dict()
    except: pass
    
    try:
        url = f'https://finance.naver.com/item/news_news.nhn?code={code}&page=1'
        res = requests.get(url, headers=HEADERS)
        res.encoding = 'EUC-KR'
        soup = BeautifulSoup(res.text, 'html.parser')
        data["뉴스"] = [a.get_text(strip=True) for a in soup.select('.title a')[:5]]
    except: pass
    
    return data

def get_yahoo_data(ticker):
    data = {"시장": "USA"}
    stock = yf.Ticker(ticker)
    
    try:
        info = stock.info
        data["주가"] = f"${info.get('currentPrice')}"
        data["기업명"] = info.get('longName')
        data["사업요약"] = info.get('longBusinessSummary')
        
        hist = stock.history(period="2y")
        if not hist.empty:
            data["기술적_지표"] = add_technical_indicators(hist)
    except: pass

    try:
        fin = stock.financials
        if not fin.empty:
            df = fin.iloc[:10, :3]
            df.columns = df.columns.astype(str)
            data["재무"] = df.to_dict()
    except: pass
    
    try:
        news = stock.news
        data["뉴스"] = [n['title'] for n in news[:5] if 'title' in n]
    except: pass

    try:
        inst_holders = stock.institutional_holders
        if inst_holders is not None and not inst_holders.empty:
            data["13F_대형기관_보유현황"] = inst_holders.head(5).astype(str).to_dict(orient='records')
    except Exception as e:
        data["13F_대형기관_보유현황"] = "조회 불가"

    # ★ [추가됨] 미국 주식 옵션 데이터 추가 ★
    data["옵션시장_동향"] = get_options_data(stock)
    
    return data

# --- AI 분석 로직 (옵션 프롬프트 추가) ---
def analyze_stock(name, data):
    prompt = f"""
    당신은 '가치 투자', '차트 분석', '수급 흐름', 그리고 **'옵션 시장의 심리(파생상품)'**까지 모두 꿰뚫어 보는 월스트리트 최상위 헤지펀드 매니저입니다.

    [분석 데이터]
    {json.dumps(data, ensure_ascii=False, default=str)}

    [보고서 작성 가이드]
    
    1. 🏰 **경제적 해자 및 비즈니스 (Fundamental)**
       - 핵심 경쟁력, 미래 성장성, 잠재적 리스크.

    2. 📊 **기술적 위치 및 타이밍 (Technical)**
       - 365선, 볼린저 밴드, 매물대 데이터를 근거로 현재 가격의 매력도 분석.

    3. 🐋 **스마트 머니 동향 (13F Institutional Holders)**
       - 대형 기관들의 지분 현황을 통한 수급 안정성 평가.

    4. 📉 **옵션 시장 심리 분석 (Options Market)** (※ 옵션 데이터가 있을 경우에만 작성)
       - **Put/Call Ratio (PCR)**: 시장 참여자들이 하락(Put)에 베팅하는지, 상승(Call)에 베팅하는지 탐욕/공포 상태를 분석하세요. (통상 1.0 이상은 약세/공포, 1.0 이하는 강세/탐욕)
       - **Max Pain (맥스페인)**: 마켓 메이커들의 이익이 극대화되는 '맥스페인 가격'이 현재 주가 대비 위에 있는지, 아래에 있는지 비교하여 단기적인 주가 자석 효과(끌림 현상)를 예측하세요.
       - **내재변동성(IV) 및 미결제약정**: 향후 주가의 변동폭이 클 것으로 예상되는지 분석하세요.

    5. 💡 **종합 투자 전략 (Verdict)**
       - 가치, 차트, 수급, 옵션 심리 4박자를 모두 고려하여 최종 결론을 내리세요.
       - 최종 투자 의견(Strong Buy / Buy / Hold / Sell)과 핵심 이유를 요약하세요.

    반드시 마크다운(Markdown) 형식을 사용하여 전문적이고 가독성 있게 작성하세요.
    """
    return model.generate_content(prompt).text

# --- 📱 화면 구성 (UI) ---
st.title("🦅 AI 주식 비서 (가치+차트+수급+옵션)")
st.markdown("재무제표, 차트, 13F 기관 수급에 이어 **옵션 시장의 맥스페인(Max Pain)과 풋콜비율**까지 통합 분석합니다.")

query = st.text_input("분석할 기업명 또는 티커 (예: 삼성전자, NVDA, TSLA)", placeholder="입력 후 엔터...")

if st.button("월스트리트급 분석 시작 🚀"):
    if not query:
        st.warning("기업 이름을 입력해주세요!")
    else:
        with st.spinner(f"🤖 '{query}'의 해자, 차트, 기관 수급, 옵션 체인까지 전부 분석 중..."):
            final_data = {}
            
            if re.search('[가-힣]', query):
                code = get_kr_stock_code(query)
                if code:
                    st.success(f"한국 주식: {query} (※ 13F 및 옵션 데이터는 미국 주식 전용입니다)")
                    final_data = get_naver_data(code)
                else:
                    st.error("종목을 찾을 수 없습니다.")
            else:
                st.info(f"미국 주식: {query.upper()}")
                final_data = get_yahoo_data(query)
            
            if final_data:
                try:
                    result = analyze_stock(query, final_data)
                    st.divider()
                    st.markdown(result)
                except Exception as e:
                    st.error(f"분석 중 오류 발생: {e}")
