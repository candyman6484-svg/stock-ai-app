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
import io

# -----------------------------------------------------------
# [설정] 페이지 기본 설정
# -----------------------------------------------------------
st.set_page_config(
    page_title="AI 주식 비서 (무결점 방어판)",
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
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept-Language': 'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7'
}

# --- 1. 기술적 지표 계산 함수 (결측치 방어) ---
def add_technical_indicators(df):
    if len(df) < 20: return {"오류": "차트 데이터 20일 미만"}
    info = {}
    
    try:
        if len(df) >= 365:
            ma365 = float(df['Close'].rolling(window=365).mean().iloc[-1])
            info['365일_이동평균선'] = int(ma365) if not np.isnan(ma365) else "계산 불가"
            info['365일선_위치'] = "365일선 상회 (장기상승세)" if df['Close'].iloc[-1] > ma365 else "365일선 하회 (하락세/저평가)"
        else:
            info['365일_이동평균선'] = "데이터 부족"

        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['Std'] = df['Close'].rolling(window=20).std()
        df['Upper'] = df['MA20'] + (df['Std'] * 2)
        df['Lower'] = df['MA20'] - (df['Std'] * 2)
        
        current_price = df['Close'].iloc[-1]
        upper = df['Upper'].iloc[-1]
        lower = df['Lower'].iloc[-1]
        
        info['볼린저밴드_상단'] = int(upper) if not np.isnan(upper) else 0
        info['볼린저밴드_하단'] = int(lower) if not np.isnan(lower) else 0
        
        if current_price >= info.get('볼린저밴드_상단', float('inf')): info['볼린저밴드_상태'] = "과매수 구간 (단기 고점)"
        elif current_price <= info.get('볼린저밴드_하단', 0): info['볼린저밴드_상태'] = "과매도 구간 (단기 저점/반등기대)"
        else: info['볼린저밴드_상태'] = "밴드 내 등락 중"

        one_year_df = df[-250:].copy()
        bins = np.linspace(one_year_df['Low'].min(), one_year_df['High'].max(), 20)
        one_year_df['PriceBin'] = pd.cut(one_year_df['Close'], bins)
        max_vol_bin = one_year_df.groupby('PriceBin', observed=False)['Volume'].sum().idxmax()
        
        info['최대_매물대_가격구간'] = str(max_vol_bin)
        if current_price < max_vol_bin.mid * 0.97: info['매물대_분석'] = "주가 위에 두터운 매물벽 (저항)"
        elif current_price > max_vol_bin.mid * 1.03: info['매물대_분석'] = "주가가 매물벽 돌파 (지지선)"
        else: info['매물대_분석'] = "최대 매물대에서 힘겨루기 중"
    except Exception as e:
        info["계산오류"] = str(e)
    return info

# --- 2. 옵션 데이터 계산 함수 (NaN 완전 제거) ---
def get_options_data(stock):
    try:
        options = stock.options
        if not options or len(options) == 0: return "옵션 만기일 데이터 없음"
        
        chain = stock.option_chain(options[0])
        # [핵심 방어] NaN 값을 0으로 완전히 치환하여 에러 방지
        calls = chain.calls.fillna(0)
        puts = chain.puts.fillna(0)
        
        if calls.empty and puts.empty: return "콜/풋 체인 데이터 없음"
        
        call_oi = float(calls['openInterest'].sum())
        put_oi = float(puts['openInterest'].sum())
        total_oi = int(call_oi + put_oi)
        pc_ratio = round(put_oi / call_oi, 2) if call_oi > 0 else 0
        
        # Max Pain 계산 (안전장치 추가)
        strikes = set(calls['strike']).union(set(puts['strike']))
        max_pain = "계산 불가"
        min_loss = float('inf')
        
        for strike in strikes:
            call_loss = calls[calls['strike'] < strike]['openInterest'] * (strike - calls[calls['strike'] < strike]['strike'])
            put_loss = puts[puts['strike'] > strike]['openInterest'] * (puts[puts['strike'] > strike]['strike'] - strike)
            total_loss = call_loss.sum() + put_loss.sum()
            
            if total_loss < min_loss:
                min_loss = total_loss
                max_pain = strike

        # IV 계산
        current_price = stock.history(period="1d")['Close'].iloc[-1]
        atm_iv = 0
        if not calls.empty and not puts.empty:
            atm_call = calls.iloc[(calls['strike'] - current_price).abs().argsort()[:1]]
            atm_put = puts.iloc[(puts['strike'] - current_price).abs().argsort()[:1]]
            if not atm_call.empty and not atm_put.empty:
                cv = float(atm_call['impliedVolatility'].values[0])
                pv = float(atm_put['impliedVolatility'].values[0])
                atm_iv = round(((cv + pv) / 2) * 100, 2)

        return {
            "분석만기일": str(options[0]),
            "풋콜비율(PCR)": pc_ratio,
            "미결제약정(OI)": total_oi,
            "맥스페인": float(max_pain) if isinstance(max_pain, (int, float)) else max_pain,
            "내재변동성(IV)": f"{atm_iv}%"
        }
    except Exception as e:
        return f"옵션 데이터 수집 오류 ({str(e)})"

# --- 3. 데이터 수집 함수 (한국주식 네이버 방어력 강화) ---
@st.cache_data(ttl=3600)
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
    except Exception as e: data["차트오류"] = str(e)

    try:
        url = f'https://finance.naver.com/item/main.naver?code={code}'
        res = requests.get(url, headers=HEADERS)
        res.encoding = 'EUC-KR'
        
        # [핵심 방어] 무식하게 테이블을 전부 찾아서 매출액이 있는 표만 추출
        dfs = pd.read_html(io.StringIO(res.text))
        target_df = None
        for d in dfs:
            if '매출액' in str(d.values) or '매출액' in str(d.columns):
                target_df = d
                break
                
        if target_df is not None:
            if isinstance(target_df.columns, pd.MultiIndex):
                target_df.columns = target_df.columns.droplevel([0, 1])
            target_df.set_index(target_df.columns[0], inplace=True)
            data["재무"] = target_df.iloc[:, :4].fillna(0).to_dict()
        else:
            data["재무"] = "재무제표를 파싱할 수 없습니다."
    except Exception as e: 
        data["재무"] = f"재무 데이터 수집 오류 ({str(e)})"
    
    try:
        soup = BeautifulSoup(res.text, 'html.parser')
        # 네이버 뉴스 태그 유연하게 모두 가져오기
        news_tags = soup.select('.title a, .tit a, .news_area a')
        news_list = [a.get_text(strip=True) for a in news_tags if a.get_text(strip=True)]
        data["뉴스"] = list(dict.fromkeys(news_list))[:5] if news_list else "최신 뉴스 없음"
    except Exception as e: 
        data["뉴스"] = f"뉴스 수집 오류 ({str(e)})"
    
    return data

# --- 4. 미국 주식 수집 ---
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
            data["재무"] = df.fillna(0).to_dict()
    except: pass

    try:
        inst_holders = stock.institutional_holders
        if isinstance(inst_holders, pd.DataFrame) and not inst_holders.empty:
            data["13F_대형기관_보유현황"] = inst_holders.head(5).fillna("").astype(str).to_dict(orient='records')
        else:
            data["13F_대형기관_보유현황"] = "최신 공시 데이터 없음"
    except Exception as e: 
        data["13F_대형기관_보유현황"] = f"수집 오류 ({str(e)})"

    data["옵션시장_동향"] = get_options_data(stock)
    
    return data

# --- AI 분석 로직 ---
def analyze_stock(name, data):
    # [핵심 방어] 제미나이 AI가 에러를 뿜지 못하도록 NaN 텍스트 강제 치환
    data_str = json.dumps(data, ensure_ascii=False, default=str)
    data_str = data_str.replace("NaN", '"데이터없음"').replace("Infinity", '"계산불가"')
    
    prompt = f"""
    당신은 '가치 투자', '차트 분석', '수급 흐름', 그리고 '옵션 시장의 심리'를 통달한 월스트리트 헤지펀드 매니저입니다.

    [분석 데이터]
    {data_str}

    [보고서 작성 가이드]
    1. 🏰 **경제적 해자 (Fundamental)**: 경쟁력, 미래 성장성, 리스크.
    2. 📊 **차트 및 타이밍 (Technical)**: 365선, 볼린저 밴드, 매물대 기반 가격 분석.
    3. 🐋 **스마트 머니 (13F 수급)**: 대형 기관 지분 현황. (미국만 해당)
    4. 📉 **옵션 시장 (Options Market)**: PCR 지수, 맥스페인 가격을 활용한 변동성 예측. (미국만 해당)
    5. 💡 **종합 전략 (Verdict)**: 최종 결론 (Strong Buy / Buy / Hold / Sell).

    마크다운(Markdown) 형식을 사용하여 전문적으로 작성하세요.
    """
    return model.generate_content(prompt).text

# --- 📱 화면 구성 (UI) ---
st.title("🦅 AI 주식 비서 (통합 방어판)")
st.markdown("가치, 차트, 수급, 옵션(맥스페인) 4박자를 안전하게 분석합니다.")

query = st.text_input("분석할 기업명/티커 (예: 카카오, TSLA)", placeholder="입력 후 엔터...")

if st.button("월스트리트급 분석 시작 🚀"):
    if not query:
        st.warning("기업 이름을 입력해주세요!")
    else:
        with st.spinner(f"🤖 '{query}' 분석 데이터를 추출 중입니다..."):
            final_data = {}
            if re.search('[가-힣]', query):
                code = get_kr_stock_code(query)
                if code:
                    st.success(f"한국 주식 감지: {query} (※ 옵션/13F 데이터는 미국 전용)")
                    final_data = get_naver_data(code)
                else: st.error("종목을 찾을 수 없습니다. (정확한 이름을 입력하세요)")
            else:
                st.info(f"미국 주식 감지: {query.upper()}")
                final_data = get_yahoo_data(query)
            
            if final_data:
                try:
                    result = analyze_stock(query, final_data)
                    st.divider()
                    st.markdown(result)
                except Exception as e:
                    st.error(f"AI 분석 생성 중 오류 발생: {e}")
