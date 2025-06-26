# app.py

# ✅ 1. 라이브러리 불러오기
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from prophet import Prophet
import io
import datetime
import numpy as np

# Google Fonts Noto Sans KR 적용 (웹페이지 기본 텍스트용)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;700&display=swap');
html, body, [class*="st-"], [class*="css-"]  {
   font-family: 'Noto Sans KR', sans-serif;
}
</style>
""", unsafe_allow_html=True)

# Matplotlib 한글 폰트 설정 (그래프용)
try:
    plt.rc('font', family='NanumGothic')
except:
    try:
        plt.rc('font', family='Malgun Gothic') # Windows
    except:
        try:
            plt.rc('font', family='AppleGothic') # Mac
        except:
            pass # 폰트가 없어도 앱은 실행되도록 함
plt.rcParams['axes.unicode_minus'] = False # 마이너스 기호 깨짐 방지


# ==============================================================================
# 💻 웹 애플리케이션 UI 구성 (✨수정된 부분✨)
# ==============================================================================
st.title("💊 의약품 재고 예측 및 관리 시스템")
st.write("지정한 기간의 데이터를 학습하여 미래 처방량을 예측하고, 현재 재고의 소진 시점을 분석합니다.")

st.sidebar.header("⚙️ 분석 설정")
csv_file = st.sidebar.file_uploader("진료 내역 데이터 (CSV)", type="csv")
xlsx_file = st.sidebar.file_uploader("의약품 정보 (XLSX)", type="xlsx")
target_code_input = st.sidebar.text_input("분석할 의약품 연합회코드 입력", "645902470")

st.sidebar.subheader("3. 예측 시나리오 설정")
train_start_date = st.sidebar.date_input("학습 시작일", datetime.date(2023, 1, 1))
train_end_date = st.sidebar.date_input("학습 종료일 (이 날짜를 기준으로 예측)", datetime.date(2023, 12, 31))

# --- ✨ 현재 재고량 입력 기능 추가 ✨ ---
st.sidebar.subheader("4. 재고 분석 설정")
current_stock = st.sidebar.number_input("현재 재고량 입력", min_value=0, value=100)
forecast_period = 180 # 재고 분석을 위해 예측 기간을 6개월로 고정

run_button = st.sidebar.button("🚀 분석 실행")

# ==============================================================================
# 📈 예측 및 시각화 실행 (✨수정된 부분✨)
# ==============================================================================
if run_button:
    if csv_file and xlsx_file and target_code_input:
        if train_start_date > train_end_date:
            st.sidebar.error("오류: 학습 종료일은 시작일보다 빠를 수 없습니다.")
        else:
            with st.spinner('데이터를 처리하고 모델을 학습하는 중입니다... 잠시만 기다려 주세요.'):
                try:
                    # --- 데이터 불러오기 및 전처리 ---
                    df = pd.read_csv(csv_file, encoding='cp949', low_memory=False)
                    name_map_df = pd.read_excel(xlsx_file)
                    name_map = dict(zip(name_map_df['연합회코드'].astype(str).str.strip(), name_map_df['연합회전용명'].astype(str).str.strip()))
                    df['진료일시'] = df['진료일시'].astype(str)
                    df['일자'] = pd.to_datetime(df['진료일시'].str[:10], errors='coerce')
                    df_valid = df[df['일자'].notna()].copy()
                    target_code = target_code_input.strip()
                    drug_name = name_map.get(target_code, f"[{target_code}]")
                    st.success(f"분석 대상 의약품: **{drug_name}**")

                    if target_code not in df_valid.columns:
                         st.error(f"입력하신 코드 '{target_code}'가 데이터 파일의 컬럼에 존재하지 않습니다.")
                    else:
                        df_valid[target_code] = pd.to_numeric(df_valid[target_code], errors='coerce').fillna(0)
                        daily_sum = df_valid.groupby('일자')[target_code].sum()
                        daily_sum = daily_sum[daily_sum > 0]
                        df_prophet_full = daily_sum.reset_index()
                        df_prophet_full.columns = ['ds', 'y']

                        start_date_dt = pd.to_datetime(train_start_date)
                        end_date_dt = pd.to_datetime(train_end_date)
                        df_prophet_train = df_prophet_full[(df_prophet_full['ds'] >= start_date_dt) & (df_prophet_full['ds'] <= end_date_dt)]
                        
                        if df_prophet_train.empty:
                            st.error(f"선택하신 기간에 처방 기록이 없습니다.")
                        else:
                            # --- 모델 학습 및 예측 ---
                            model = Prophet(daily_seasonality=True)
                            model.fit(df_prophet_train)
                            future = model.make_future_dataframe(periods=forecast_period, freq='D')
                            forecast = model.predict(future)
                            
                            # --- ✨ 재고 소진일 계산 로직 ✨ ---
                            # 예측 결과에서 미래 부분만 추출
                            future_fc = forecast[forecast['ds'] > end_date_dt].copy()
                            # 예측 처방량이 음수일 경우 0으로 처리
                            future_fc['yhat'] = future_fc['yhat'].clip(lower=0)
                            # 날짜별 누적 처방량 계산
                            future_fc['cumulative_yhat'] = future_fc['yhat'].cumsum()
                            
                            # 누적 처방량이 현재 재고량을 초과하는 첫 날 찾기
                            stock_out_day = future_fc[future_fc['cumulative_yhat'] >= current_stock]

                            # --- ✨ 결과 텍스트 출력 ✨ ---
                            st.subheader("📦 재고 분석 결과")
                            col1, col2 = st.columns(2)
                            col1.metric("현재 재고량", f"{current_stock} 개")

                            if not stock_out_day.empty:
                                stock_out_date = stock_out_day.iloc[0]['ds']
                                days_left = (stock_out_date - end_date_dt).days
                                col2.metric("재고 소진까지 남은 기간", f"약 {days_left} 일", f"예상 소진일: {stock_out_date.strftime('%Y-%m-%d')}")
                                st.success(f"**분석 요약:** 현재 재고({current_stock}개)는 앞으로 **약 {days_left}일** 후인 **{stock_out_date.strftime('%Y-%m-%d')}** 경에 소진될 것으로 예측됩니다.")
                            else:
                                col2.metric("재고 소진까지 남은 기간", f"{forecast_period} 일 이상")
                                st.info(f"**분석 요약:** 현재 재고({current_stock}개)는 예측 기간인 **{forecast_period}일** 내에는 소진되지 않을 것으로 보입니다.")


                            # --- ✨ 그래프 시각화 ✨ ---
                            st.subheader(f"📊 {train_start_date.strftime('%Y-%m-%d')} ~ {train_end_date.strftime('%Y-%m-%d')} 데이터 학습 결과 및 예측")
                            fig, ax = plt.subplots(figsize=(14, 7))
                            history_fc = forecast[forecast['ds'] <= end_date_dt]
                            
                            ax.plot(history_fc['ds'], history_fc['yhat'], color='gray', linestyle='-', linewidth=1.5, label='과거 데이터 모델 적합')
                            ax.plot(future_fc['ds'], future_fc['yhat'], color='#0072B2', linestyle='-', linewidth=2, label='미래 예측')
                            ax.fill_between(future_fc['ds'], future_fc['yhat_lower'].clip(lower=0), future_fc['yhat_upper'], color='#0072B2', alpha=0.2)
                            ax.plot(df_prophet_train['ds'], df_prophet_train['y'], 'k.', markersize=4, label='실제 처방량')
                            ax.axvline(x=end_date_dt, color='red', linestyle='--', linewidth=1.5, label='예측 시작일')
                            
                            # 재고 소진일에 마커 표시
                            if not stock_out_day.empty:
                                ax.axvline(x=stock_out_date, color='darkorange', linestyle=':', linewidth=2, label=f'재고 소진 예상일 ({days_left}일 후)')

                            ax.set_title(f"{drug_name} ({target_code}) 처방량 예측", fontsize=16)
                            ax.set_xlabel("날짜", fontsize=12)
                            ax.set_ylabel("처방 수량", fontsize=12)
                            ax.legend()
                            ax.grid(True, which='major', c='gray', ls='-', lw=1, alpha=0.2)
                            fig.autofmt_xdate()
                            st.pyplot(fig)
                            
                            st.subheader("🔬 지정 기간 데이터의 패턴 분석")
                            fig2 = model.plot_components(forecast)
                            st.pyplot(fig2)

                except Exception as e:
                    st.error(f"오류가 발생했습니다: {e}")
    else:
        st.warning("모든 파일을 업로드하고 약물 코드를 입력한 후 버튼을 눌러주세요.")
