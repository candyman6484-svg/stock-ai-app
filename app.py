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
    page_title="AI 주식 비서 (통합 분석판)",
    page_icon="🔮",
    layout="centered"
)

# -----------------------------------------------------------
# [설정] API 키 (Secrets에서 가져오기)
# -----------------------------------------------------------
try:
    if "GOOGLE_API_KEY" in st.secrets:
        GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    else:
        # 로컬 테스트용
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

# --- 기술적 지표 계산 함수 (그대로 유지) ---

def add_technical_indicators(df):
    if len(df) < 20: return {}

    info = {}
    
    # 1. 365일 이동평균선
    if len(df) >= 365:
        ma365 = df['Close'].rolling(window=365).mean().iloc[-1]
        info['365일_이동평균선'] = int(ma365)
        info['365일선_위치'] = "주가가 365일선 위에 있음 (장기상승세)" if df['Close'].iloc[-1] > ma365 else "주가가 365일선 아래에 있음 (장기하락세/저평가)"
    else:
        info['365일_이동평균선'] = "데이터 부족"

    # 2. 볼린저 밴드
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
        info['볼린저밴드_상태'] = "과매수 구간 (단기 고점)"
    elif current_price <= lower:
        info['볼린저밴드_상태'] = "과매도 구간 (단기 저점/반등기대)"
    else:
        info['볼린저밴드_상태'] = "밴드 내 등락 중"

    # 3. 매물대
    one_year_df = df[-250:] 
    price_min = one_year_df['Low'].min()
    price_max = one_year_df['High'].max()
    bins = np.linspace(price_min, price_max, 20)
    
    one_year_df['PriceBin'] = pd.cut(one_year_df['Close'], bins)
    volume_profile = one_year_df.groupby('PriceBin')['Volume'].sum()
    
    max_vol_bin = volume_profile.idxmax()
    info['최대_매물대_가격구간'] = str(max_vol_bin)
    
    mid_point_resistance = max_vol_bin.mid
    if current_price < mid_point_resistance * 0.97:
        info['매물대_분석'] = "주가 위에 두터운 매물벽 존재 (저항)"
    elif current_price > mid_point_resistance * 1.03:
        info['매물대_분석'] = "주가가 매물벽을 뚫고 지지받는 중"
    else:
        info['매물대_분석'] = "최대 매물대 구간에서 힘겨루기 중"

    return info

# --- 데이터 수집 함수 ---

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
            cur = df.iloc[-1]
            data["주가"] = f"{int(cur['Close']):,}원"
            data["기술적_지표"] = add_technical_indicators(df)
    except Exception as e: data["차트오류"] = str(e)

    try:
        url = f'https://finance.naver.com/item/main.naver?code={code}'
        res = requests.get(url, headers=HEADERS)
        res.encoding = 'EUC-KR'
        dfs = pd.read_html(res.text, match='매출액')
        if dfs:
            df = dfs[0].set_index(dfs[0].columns[0])
            data["재무"] = df.iloc[:, :4].to_dict()
    except: pass
    
    # 뉴스 추가 (정성적 분석의 핵심 재료)
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
        data["사업요약"] = info.get('longBusinessSummary') # 사업 내용 추가
        
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
    
    return data

# --- [핵심 수정] AI 분석 로직 (하이브리드 프롬프트) ---

def analyze_stock(name, data):
    prompt = f"""
    당신은 '워렌 버핏의 가치 투자 철학'과 '월스트리트의 기술적 분석'을 모두 통달한 최고의 투자 전략가입니다.
    제공된 데이터(재무, 뉴스, 차트 지표)를 종합하여 **'좋은 기업을(Qualitative) 좋은 가격에(Technical) 살 수 있는지'** 심층 분석하세요.

    [분석 데이터]
    {json.dumps(data, ensure_ascii=False, default=str)}

    [보고서 작성 가이드]
    
    1. 🏰 **경제적 해자 및 비즈니스 분석 (Fundamental)**
       - **핵심 경쟁력**: 이 기업이 경쟁사들이 넘볼 수 없는 기술력, 브랜드, 네트워크 효과를 가졌는지 분석하세요.
       - **미래 성장성**: AI, 친환경 등 미래 산업 트렌드와 이 기업이 어떻게 연결되는지 설명하세요.
       - **잠재적 리스크**: 뉴스 데이터를 참고하여 경영진 리스크, 규제, 경쟁 심화 등 '악재'를 냉정하게 평가하세요.

    2. 📊 **기술적 분석 및 타이밍 (Technical)**
       - **추세 판단**: 365일 이동평균선을 기준으로 현재가 장기 상승세인지 하락세인지 판단하세요.
       - **매물대 분석**: 현재 주가 주변에 강력한 '매물벽(저항)'이 있는지, 아니면 '지지선'이 받쳐주는지 분석하세요.
       - **과열 여부**: 볼린저 밴드를 기준으로 지금 사는 것이 너무 비싼지(과매수), 싼지(과매도) 평가하세요.

    3. 💡 **종합 투자 판단 (Verdict)**
       - **기업 점수**: "이 기업은 10년 뒤에도 살아남을 위대한 기업인가?" (100점 만점)
       - **타이밍 점수**: "지금이 매수하기 좋은 가격대인가?" (100점 만점)
       - **최종 전략**: (Strong Buy / Buy / Hold / Sell) 중 하나를 선택하고, 그 이유를 한 문장으로 요약하세요.
         (예: "기업 가치는 훌륭하나(90점), 차트상 단기 과열이므로(40점) 조정 시 분할 매수 추천")

    반드시 마크다운(Markdown) 형식을 사용하여 가독성 있게 작성하세요.
    """
    return model.generate_content(prompt).text

# --- 📱 화면 구성 (UI) ---

st.title("🔮 AI 주식 비서 (가치+차트)")
st.markdown("워렌 버핏의 눈으로 **기업**을 보고, 트레이더의 눈으로 **타이밍**을 봅니다.")

query = st.text_input("분석할 기업명 또는 티커 (예: 삼성전자, NVDA)", placeholder="입력 후 엔터...")

if st.button("통합 분석 시작 🚀"):
    if not query:
        st.warning("기업 이름을 입력해주세요!")
    else:
        with st.spinner(f"🤖 '{query}'의 경제적 해자와 차트를 동시에 분석 중..."):
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
