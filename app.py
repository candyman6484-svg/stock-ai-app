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
import io  # 네이버 웹 크롤링 우회용 도구 추가

# -----------------------------------------------------------
# [설정] 페이지 기본 설정
# -----------------------------------------------------------
st.set_page_config(
    page_title="AI 주식 비서 (오류 방어 적용판)",
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

# 네이버 차단 우회용 초강력 헤더
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
    'Accept-Language': 'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7'
}

# --- 기술적 지표 계산 함수 ---
def add_technical_indicators(df):
    if len(df) < 20: return {}
    info = {}
    
    if len(df) >= 365:
        ma365 = df['Close'].rolling(window=365).mean().iloc[-1]
        info['365일_이동평균선'] = int(ma365)
        info['365일선_위치'] = "365일선 상회 (장기상승세)" if df['Close'].iloc[-1] > ma365 else "365일선 하회 (하락세/저평가)"
    else:
        info['365일_이동평균선'] = "데이터 부족"

    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['Std'] = df['Close'].rolling(window=20).std()
    df['Upper'] = df['MA20'] + (df['Std'] * 2)
    df['Lower'] = df['MA20'] - (df['Std'] * 2)
    
    current_price = df['Close'].iloc[-1]
    info['볼린저밴드_상단'] = int(df['Upper'].iloc[-1])
    info['볼린저밴드_하단'] = int(df['Lower'].iloc[-1])
    
    if current_price >= info['볼린저밴드_상단']: info['볼린저밴드_상태'] = "과매수 구간 (단기 고점)"
    elif current_price <= info['볼린저밴드_하단']: info['볼린저밴드_상태'] = "과매도 구간 (단기 저점/반등기대)"
    else: info['볼린저밴드_상태'] = "밴드 내 등락 중"

    one_year_df = df[-250:] 
    bins = np.linspace(one_year_df['Low'].min(), one_year_df['High'].max(), 20)
    one_year_df['PriceBin'] = pd.cut(one_year_df['Close'], bins)
    max_vol_bin = one_year_df.groupby('PriceBin')['Volume'].sum().idxmax()
    
    info['최대_매물대_가격구간'] = str(max_vol_bin)
    if current_price < max_vol_bin.mid * 0.97: info['매물대_분석'] = "주가 위에 두터운 매물벽 (저항)"
    elif current_price > max_vol_bin.mid * 1.03: info['매물대_분석'] = "주가가 매물벽 돌파 (지지선)"
    else: info['매물대_분석'] = "최대 매물대에서 힘겨루기 중"
    return info

# --- [수정] 옵션 데이터 계산 함수 (에러 방어 적용) ---
def get_options_data(stock):
    try:
        options = stock.options
        if not options: return "옵션 데이터 제공 안 함"
        
        nearest_expiry = options[0]
        chain = stock.option_chain(nearest_expiry)
        calls, puts = chain.calls, chain.puts
        
        if calls.empty and puts.empty: return "체인 데이터 없음"
        
        # [핵심] 비어있는 값(NaN)을 0으로 강제 변환하여 계산 오류 원천 차단
        calls['openInterest'] = calls['openInterest'].fillna(0)
        puts['openInterest'] = puts['openInterest'].fillna(0)
        calls['impliedVolatility'] = calls['impliedVolatility'].fillna(0)
        puts['impliedVolatility'] = puts['impliedVolatility'].fillna(0)
        
        call_oi = calls['openInterest'].sum()
        put_oi = puts['openInterest'].sum()
        total_oi = call_oi + put_oi
        pc_ratio = round(put_oi / call_oi, 2) if call_oi > 0 else 0
        
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

        current_price = stock.history(period="1d")['Close'].iloc[-1]
        atm_call = calls.iloc[(calls['strike'] - current_price).abs().argsort()[:1]]
        atm_put = puts.iloc[(puts['strike'] - current_price).abs().argsort()[:1]]
        
        atm_iv = 0
        if not atm_call.empty and not atm_put.empty:
            atm_iv = (atm_call['impliedVolatility'].values[0] + atm_put['impliedVolatility'].values[0]) / 2
            atm_iv = round(atm_iv * 100, 2)

        return {
            "분석만기일": nearest_expiry,
            "풋콜비율(PCR)": pc_ratio,
            "미결제약정(OI)": int(total_oi),
            "맥스페인": max_pain,
            "내재변동성(IV)": f"{atm_iv}%"
        }
    except Exception as e:
        return f"옵션 데이터 일시적 수집 오류 ({str(e)})"

# --- [수정] 데이터 수집 함수 (한국주식 네이버 방어) ---
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
        # [핵심] io.StringIO()를 추가하여 최신 Pandas 버그 우회 및 크롤링 강화
        dfs = pd.read_html(io.StringIO(res.text), match='매출액')
        if dfs:
            df = dfs[0].set_index(dfs[0].columns[0])
            data["재무"] = df.iloc[:, :4].to_dict()
    except Exception as e: 
        data["재무"] = f"재무 데이터 수집 오류 (네이버 차단 등)"
    
    try:
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
        if not hist.empty: data["기술적_지표"] = add_technical_indicators(hist)
    except: pass

    try:
        fin = stock.financials
        if not fin.empty:
            df = fin.iloc[:10, :3]
            df.columns = df.columns.astype(str)
            data["재무"] = df.to_dict()
    except: pass

    # [수정] 13F 기관 보유 현황 방어 로직
    try:
        inst_holders = stock.institutional_holders
        if inst_holders is not None and not inst_holders.empty:
            data["13F_대형기관_보유현황"] = inst_holders.head(5).astype(str).to_dict(orient='records')
        else:
            data["13F_대형기관_보유현황"] = "최신 공시 데이터 없음 (야후 서버 이슈 가능성)"
    except: 
        data["13F_대형기관_보유현황"] = "수집 불가"

    data["옵션시장_동향"] = get_options_data(stock)
    
    return data

# --- AI 분석 로직 (유지) ---
def analyze_stock(name, data):
    prompt = f"""
    당신은 '가치 투자', '차트 분석', '수급 흐름', 그리고 '옵션 시장의 심리(파생상품)'까지 모두 꿰뚫어 보는 월스트리트 최상위 헤지펀드 매니저입니다.

    [분석 데이터]
    {json.dumps(data, ensure_ascii=False, default=str)}

    [보고서 작성 가이드]
    1. 🏰 **경제적 해자 및 비즈니스 (Fundamental)**: 경쟁력, 미래 성장성, 잠재적 리스크.
    2. 📊 **기술적 위치 및 타이밍 (Technical)**: 365선, 볼린저 밴드, 매물대 근거 가격 매력도.
    3. 🐋 **스마트 머니 동향 (13F Institutional Holders)**: 대형 기관 지분 현황 및 수급 안정성.
    4. 📉 **옵션 시장 심리 분석 (Options Market)**: PCR 지수 비교, 맥스페인 가격과 현재 주가 비교를 통한 자석 효과 분석, 내재변동성 평가. (데이터 있을 시)
    5. 💡 **종합 투자 전략 (Verdict)**: 가치, 차트, 수급, 옵션 심리 4박자 고려 최종 결론 (Strong Buy / Buy / Hold / Sell).

    마크다운(Markdown) 형식을 사용하여 작성하세요.
    """
    return model.generate_content(prompt).text

# --- 📱 화면 구성 (UI) ---
st.title("🦅 AI 주식 비서 (가치+차트+수급+옵션)")
st.markdown("재무, 차트, 수급은 물론 **옵션 시장의 맥스페인**까지 분석합니다.")

query = st.text_input("분석할 기업명 또는 티커 (예: 삼성전자, NVDA)", placeholder="입력 후 엔터...")

if st.button("월스트리트급 분석 시작 🚀"):
    if not query:
        st.warning("기업 이름을 입력해주세요!")
    else:
        with st.spinner(f"🤖 '{query}'의 데이터를 수집하고 분석 중입니다..."):
            final_data = {}
            if re.search('[가-힣]', query):
                code = get_kr_stock_code(query)
                if code:
                    st.success(f"한국 주식: {query} (※ 옵션/13F 데이터는 미국 전용)")
                    final_data = get_naver_data(code)
                else: st.error("종목을 찾을 수 없습니다.")
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
