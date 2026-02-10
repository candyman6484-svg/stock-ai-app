import streamlit as st
import google.generativeai as genai
import FinanceDataReader as fdr
import yfinance as yf
import requests
import pandas as pd
import numpy as np # 수학 계산용
from bs4 import BeautifulSoup
import re
import json

# -----------------------------------------------------------
# [설정] 페이지 기본 설정
# -----------------------------------------------------------
st.set_page_config(
    page_title="AI 주식 비서 (기술적 분석 추가)",
    page_icon="📈",
    layout="centered"
)

# -----------------------------------------------------------
# [설정] API 키 (Secrets에서 가져오기)
# -----------------------------------------------------------
try:
    if "GOOGLE_API_KEY" in st.secrets:
        GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    else:
        # 로컬 테스트용 (본인 키 입력)
        GOOGLE_API_KEY = "여기에_API키를_직접_넣으세요_로컬테스트시"

    clean_key = re.sub(r'[^a-zA-Z0-9_\-]', '', GOOGLE_API_KEY)
    genai.configure(api_key=clean_key)
    model = genai.GenerativeModel('gemini-3-flash-preview')
except Exception as e:
    st.error(f"API 키 설정 오류: {e}")

HEADERS = {
    'User-Agent': 'Mozilla/5.0',
    'Accept-Language': 'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7'
}

# --- [핵심] 기술적 지표 계산 함수 (새로 추가됨!) ---

def add_technical_indicators(df):
    """
    주가 데이터(df)를 받아서 365선, 볼린저밴드, 매물대를 계산해줍니다.
    """
    if len(df) < 20: return {} # 데이터가 너무 적으면 계산 불가

    info = {}
    
    # 1. 365일 이동평균선 (1년 추세선)
    if len(df) >= 365:
        ma365 = df['Close'].rolling(window=365).mean().iloc[-1]
        info['365일_이동평균선'] = int(ma365)
        info['365일선_위치'] = "주가가 365일선 위에 있음 (장기상승세)" if df['Close'].iloc[-1] > ma365 else "주가가 365일선 아래에 있음 (장기하락세/저평가)"
    else:
        info['365일_이동평균선'] = "데이터 부족으로 계산 불가 (상장 1년 미만)"

    # 2. 볼린저 밴드 (20일 기준)
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['Std'] = df['Close'].rolling(window=20).std()
    df['Upper'] = df['MA20'] + (df['Std'] * 2)
    df['Lower'] = df['MA20'] - (df['Std'] * 2)
    
    current_price = df['Close'].iloc[-1]
    upper = df['Upper'].iloc[-1]
    lower = df['Lower'].iloc[-1]
    
    info['볼린저밴드_상단'] = int(upper)
    info['볼린저밴드_하단'] = int(lower)
    
    if current_price >= upper:
        info['볼린저밴드_상태'] = "과매수 구간 (상단 돌파, 조정 주의)"
    elif current_price <= lower:
        info['볼린저밴드_상태'] = "과매도 구간 (하단 이탈, 반등 가능성)"
    else:
        info['볼린저밴드_상태'] = "밴드 내에서 등락 중 (일반적인 흐름)"

    # 3. 매물대 (최근 1년 가장 거래가 많았던 가격대)
    # 최근 1년 데이터만 자르기
    one_year_df = df[-250:] 
    
    # 가격을 20개 구간으로 나눔
    price_min = one_year_df['Low'].min()
    price_max = one_year_df['High'].max()
    bins = np.linspace(price_min, price_max, 20)
    
    # 각 구간별 거래량 합계 계산
    one_year_df['PriceBin'] = pd.cut(one_year_df['Close'], bins)
    volume_profile = one_year_df.groupby('PriceBin')['Volume'].sum()
    
    # 거래량이 가장 많은 구간(매물대) 찾기
    max_vol_bin = volume_profile.idxmax()
    info['최대_매물대_가격구간'] = str(max_vol_bin)
    
    # 현재 주가와 매물대 비교
    mid_point_resistance = max_vol_bin.mid
    if current_price < mid_point_resistance * 0.97:
        info['매물대_분석'] = "현재 주가 위에 거대한 매물벽이 있음 (상승 시 저항 예상)"
    elif current_price > mid_point_resistance * 1.03:
        info['매물대_분석'] = "현재 주가가 거대한 매물대를 뚫고 올라옴 (지지선 역할 기대)"
    else:
        info['매물대_분석'] = "현재 최대 매물대 구간에서 힘겨루기 중"

    return info

# --- 기존 데이터 수집 함수 수정 ---

def get_kr_stock_code(name):
    try:
        df = fdr.StockListing('KRX')
        matches = df[df['Name'] == name]['Code'].values
        return matches[0] if len(matches) > 0 else None
    except: return None

def get_naver_data(code):
    data = {"시장": "Korea"}
    
    # [수정] 차트 분석을 위해 2년치 데이터 가져오기
    try:
        df = fdr.DataReader(code) # 전체 데이터 가져옴
        if not df.empty:
            cur = df.iloc[-1]
            data["주가"] = f"{int(cur['Close']):,}원"
            
            # ★ 기술적 지표 계산 추가 ★
            tech_data = add_technical_indicators(df)
            data["기술적_분석_지표"] = tech_data
    except Exception as e: 
        data["기술적_분석_오류"] = str(e)

    try:
        url = f'https://finance.naver.com/item/main.naver?code={code}'
        res = requests.get(url, headers=HEADERS)
        res.encoding = 'EUC-KR'
        dfs = pd.read_html(res.text, match='매출액')
        if dfs:
            df = dfs[0].set_index(dfs[0].columns[0])
            data["재무"] = df.iloc[:, :4].to_dict()
    except: pass
    
    return data

def get_yahoo_data(ticker):
    data = {"시장": "USA"}
    stock = yf.Ticker(ticker)
    try:
        info = stock.info
        data["주가"] = f"${info.get('currentPrice')}"
        data["기업명"] = info.get('longName')
        
        # [수정] 차트 분석을 위해 2년치 데이터 가져오기 (365선 계산용)
        hist = stock.history(period="2y")
        if not hist.empty:
            # ★ 기술적 지표 계산 추가 ★
            tech_data = add_technical_indicators(hist)
            data["기술적_분석_지표"] = tech_data
            
    except: pass

    try:
        fin = stock.financials
        if not fin.empty:
            df = fin.iloc[:10, :3]
            df.columns = df.columns.astype(str)
            data["재무"] = df.to_dict()
    except: pass
    
    return data

def analyze_stock(name, data):
    # 프롬프트에 기술적 분석 내용 반영
    prompt = f"""
    당신은 30년 경력의 '차트 및 가치 투자 전문가'입니다. 
    제공된 '{name}'의 재무제표(가치)와 기술적 지표(차트)를 모두 통합하여 분석하세요.
    
    [분석 데이터]
    {json.dumps(data, ensure_ascii=False, default=str)}

    [필수 분석 항목]
    1. **기술적 위치 분석 (차트)**:
       - **365일선**: 현재 주가가 장기 추세선(365선) 위에 있는가 아래에 있는가? (추세 판단)
       - **볼린저 밴드**: 과매수(비쌈) 구간인가 과매도(쌈) 구간인가?
       - **매물대**: 머리 위에 저항벽이 있는가, 발 아래 지지선이 있는가?
       
    2. **기본적 분석 (가치)**:
       - 매출과 이익이 성장하고 있는가? 재무는 튼튼한가?

    3. **최종 투자 전략**:
       - 차트상 지금 사도 되는 타이밍인가? 아니면 기다려야 하는가?
       - 10점 만점에 점수를 매기고, 한 줄로 요약하세요.
       - (예: "가치는 훌륭하나 차트상 과열 구간이므로 조정 시 매수 추천")

    마크다운 형식으로 깔끔하게 작성하세요.
    """
    return model.generate_content(prompt).text

# --- 📱 화면 구성 (UI) ---

st.title("📈 AI 주식 비서 (PRO 버전)")
st.markdown("재무제표뿐만 아니라 **매물대, 365선, 볼린저 밴드**까지 분석합니다.")

query = st.text_input("분석할 기업명 또는 티커 (예: 삼성전자, NVDA)", placeholder="입력 후 엔터...")

if st.button("차트 & 가치 분석 시작 🚀"):
    if not query:
        st.warning("기업 이름을 입력해주세요!")
    else:
        with st.spinner(f"🤖 '{query}'의 차트와 재무를 뜯어보는 중..."):
            final_data = {}
            
            if re.search('[가-힣]', query):
                code = get_kr_stock_code(query)
                if code:
                    st.success(f"한국 주식 감지: {query} ({code})")
                    final_data = get_naver_data(code)
                else:
                    st.error("종목을 찾을 수 없습니다.")
            else:
                st.info(f"미국 주식 감지: {query.upper()}")
                final_data = get_yahoo_data(query)
            
            if final_data:
                try:
                    result = analyze_stock(query, final_data)
                    st.divider()
                    st.markdown(result)
                except Exception as e:
                    st.error(f"분석 중 오류 발생: {e}")
