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
    page_title="AI 주식 비서 (가치+차트+수급)",
    page_icon="👑",
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

# --- 기술적 지표 계산 함수 ---
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

    # ★ [추가됨] SEC 13F 기반 대형 기관 보유 현황 크롤링 ★
    try:
        inst_holders = stock.institutional_holders
        if inst_holders is not None and not inst_holders.empty:
            # 상위 5개 기관의 데이터를 가져와 문자열로 변환 (오류 방지)
            data["13F_대형기관_보유현황"] = inst_holders.head(5).astype(str).to_dict(orient='records')
    except Exception as e:
        data["13F_대형기관_보유현황"] = "조회 불가"
    
    return data

# --- AI 분석 로직 ---
def analyze_stock(name, data):
    prompt = f"""
    당신은 '워렌 버핏의 가치 투자 철학', '월스트리트의 기술적 분석', 그리고 '헤지펀드의 수급 추적'을 모두 통달한 수석 투자 전략가입니다.

    [분석 데이터]
    {json.dumps(data, ensure_ascii=False, default=str)}

    [보고서 작성 가이드]
    
    1. 🏰 **경제적 해자 및 비즈니스 (Fundamental)**
       - 핵심 경쟁력, 미래 성장성, 잠재적 리스크를 분석하세요.

    2. 📊 **기술적 위치 및 타이밍 (Technical)**
       - 365선, 볼린저 밴드, 매물대 데이터를 근거로 현재 가격의 매력도와 타이밍을 분석하세요.

    3. 🐋 **스마트 머니 동향 (13F Institutional Holders)**
       - 제공된 '13F_대형기관_보유현황' 데이터를 바탕으로 어떤 대형 기관(Vanguard, BlackRock 등)이 이 기업을 신뢰하고 보유 중인지 분석하세요. (데이터가 없다면 생략 가능)
       - 기관의 지분이 탄탄하게 받쳐주고 있는지, 수급 측면에서의 안정성을 평가하세요.

    4. 💡 **종합 투자 전략 (Verdict)**
       - 기업의 본질 가치, 차트 타이밍, 대형 기관의 수급을 종합하여 최종 결론을 내리세요.
       - 최종 투자 의견(Strong Buy / Buy / Hold / Sell)과 그 이유를 명확히 제시하세요.

    반드시 마크다운(Markdown) 형식을 사용하여 가독성 있게 작성하세요.
    """
    return model.generate_content(prompt).text

# --- 📱 화면 구성 (UI) ---
st.title("👑 AI 주식 비서 (가치+차트+수급)")
st.markdown("재무제표, 차트 분석에 이어 **13F 공시 기반 대형 기관의 움직임**까지 추적합니다.")

query = st.text_input("분석할 기업명 또는 티커 (예: 삼성전자, NVDA, PLTR)", placeholder="입력 후 엔터...")

if st.button("전문가 분석 시작 🚀"):
    if not query:
        st.warning("기업 이름을 입력해주세요!")
    else:
        with st.spinner(f"🤖 '{query}'의 해자, 차트, 기관 수급을 샅샅이 뒤지는 중..."):
            final_data = {}
            
            if re.search('[가-힣]', query):
                code = get_kr_stock_code(query)
                if code:
                    st.success(f"한국 주식: {query} (13F 수급 데이터는 미국 주식에 한함)")
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
