import streamlit as st
import google.generativeai as genai
import FinanceDataReader as fdr
import yfinance as yf
import requests
import pandas as pd
from bs4 import BeautifulSoup
import re
import json

# -----------------------------------------------------------
# [설정] 페이지 기본 설정
# -----------------------------------------------------------
st.set_page_config(
    page_title="AI 주식 비서",
    page_icon="📈",
    layout="centered"
)

# -----------------------------------------------------------
# [설정] API 키 (여기에 본인의 키를 입력하세요)
# -----------------------------------------------------------
import streamlit as st
# 스트림릿 서버의 비밀 금고(secrets)에서 키를 가져옵니다.
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]

# API 키 세탁 및 설정
try:
    if GOOGLE_API_KEY:
        clean_key = re.sub(r'[^a-zA-Z0-9_\-]', '', GOOGLE_API_KEY)
        genai.configure(api_key=clean_key)
        model = genai.GenerativeModel('gemini-3-flash-preview')
except Exception as e:
    st.error(f"API 키 설정 오류: {e}")

# 네이버 헤더
HEADERS = {
    'User-Agent': 'Mozilla/5.0',
    'Accept-Language': 'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7'
}

# --- 함수 모음 (기존 로직과 동일) ---

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
    
    return data

def get_yahoo_data(ticker):
    data = {"시장": "USA"}
    stock = yf.Ticker(ticker)
    try:
        info = stock.info
        data["주가"] = f"${info.get('currentPrice')}"
        data["기업명"] = info.get('longName')
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
    prompt = f"""
    당신은 워렌 버핏입니다. '{name}' 데이터를 보고 장기 투자 보고서를 써주세요.
    
    [데이터]
    {json.dumps(data, ensure_ascii=False, default=str)}

    [요청]
    1. 경제적 해자, 미래 성장성, 리스크를 분석하세요.
    2. 10년 뒤 전망(Strong Buy/Sell)을 내리세요.
    3. 가독성 좋게 마크다운으로 작성하세요.
    """
    return model.generate_content(prompt).text

# --- 📱 화면 구성 (UI) ---

st.title("📈 나만의 AI 주식 비서")
st.markdown("PC와 아이폰 어디서든 접속 가능한 **개인용 분석 앱**입니다.")

# 입력창
query = st.text_input("분석할 기업명 또는 티커 (예: 삼성전자, TSLA)", placeholder="입력 후 엔터...")

# 버튼
if st.button("분석 시작 🚀"):
    if not query:
        st.warning("기업 이름을 입력해주세요!")
    else:
        with st.spinner(f"🤖 '{query}' 분석 중입니다... 잠시만 기다려주세요."):
            final_data = {}
            
            # 한글 감지
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
            
            # 분석 요청
            if final_data:
                try:
                    result = analyze_stock(query, final_data)
                    st.divider()
                    st.markdown(result) # 결과 보여주기
                except Exception as e:
                    st.error(f"분석 중 오류 발생: {e}")