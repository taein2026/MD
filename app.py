# app.py

# ✅ 1. 라이브러리 불러오기
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from prophet import Prophet
import io
import datetime

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
st.title("💊 의약품 처방량 시나리오 예측")
st.write("지정한 기간의 데이터만 학습하여 미래 처방량을 예측합니다.")

st.sidebar.header("⚙️ 분석 설정")
csv_file = st.sidebar.file_uploader("진료 내역 데이터 (CSV)", type="csv")
xlsx_file = st.sidebar.file_uploader("의약품 정보 (XLSX)", type="xlsx")
target_code_input = st.sidebar.text_input("분석할 의약품 연합회코드 입력", "645902470")

st.sidebar.subheader("3. 예측 시나리오 설정")
# 학습 기간 시작일과 종료일을 모두 지정하도록 변경
train_start_date = st.sidebar.date_input("학습 시작일", datetime.date(2023, 1, 1))
train_end_date = st.sidebar.date_input("학습 종료일 (이 날짜를 기준으로 예측)", datetime.date(2023, 12, 31))
forecast_period = st.sidebar.number_input("예측 기간 (일)", min_value=1, max_value=365, value=14)

run_button = st.sidebar.button("🚀 예측 실행")

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
                    # --- 데이터 불러오기 및 전체 전처리 ---
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

                        # --- ✨ 사용자가 선택한 '학습 시작/종료일'로 데이터 필터링 ---
                        start_date_dt = pd.to_datetime(train_start_date)
                        end_date_dt = pd.to_datetime(train_end_date)
                        df_prophet_train = df_prophet_full[(df_prophet_full['ds'] >= start_date_dt) & (df_prophet_full['ds'] <= end_date_dt)]
                        
                        if df_prophet_train.empty:
                            st.error(f"선택하신 '{train_start_date.strftime('%Y-%m-%d')} ~ {train_end_date.strftime('%Y-%m-%d')}' 기간에 처방 기록이 없습니다.")
                        else:
                            # --- 필터링된 데이터로 새 모델 학습 ---
                            model = Prophet(daily_seasonality=True)
                            model.fit(df_prophet_train)

                            # --- 학습 종료일로부터 사용자 입력 기간만큼 미래 예측 ---
                            future = model.make_future_dataframe(periods=forecast_period, freq='D')
                            forecast = model.predict(future)

                            st.subheader(f"📊 {train_start_date.strftime('%Y-%m-%d')} ~ {train_end_date.strftime('%Y-%m-%d')} 데이터 학습 결과 및 예측")
                            
                            fig1 = model.plot(forecast)
                            
                            ax = fig1.gca()
                            ax.axvline(x=end_date_dt, color='red', linestyle='--', linewidth=1.5, label='예측 시작일')
                            ax.set_title(f"{drug_name} ({target_code}) 처방량 예측", fontsize=16)
                            ax.set_xlabel("날짜", fontsize=12)
                            ax.set_ylabel("처방 수량", fontsize=12)
                            
                            # 그래프 확대 (학습 시작일 부터 예측 마지막 날까지)
                            ax.set_xlim([start_date_dt, forecast['ds'].max()])
                            
                            st.pyplot(fig1)

                            st.subheader("🔬 지정 기간 데이터의 패턴 분석")
                            fig2 = model.plot_components(forecast)
                            st.pyplot(fig2)

                except Exception as e:
                    st.error(f"오류가 발생했습니다: {e}")
    else:
        st.warning("모든 파일을 업로드하고 약물 코드를 입력한 후 버튼을 눌러주세요.")
