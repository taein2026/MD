# app.py

# ✅ 1. 라이브러리 불러오기
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from prophet import Prophet
import io
import datetime

# Google Fonts Noto Sans KR 적용
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;700&display=swap');
html, body, [class*="st-"], [class*="css-"]  {
   font-family: 'Noto Sans KR', sans-serif;
}
</style>
""", unsafe_allow_html=True)

# Matplotlib 한글 폰트 설정
try:
    plt.rc('font', family='NanumGothic')
except:
    try:
        plt.rc('font', family='Malgun Gothic')
    except:
        try:
            plt.rc('font', family='AppleGothic')
        except:
            pass
plt.rcParams['axes.unicode_minus'] = False


# ==============================================================================
# 💻 웹 애플리케이션 UI 구성
# ==============================================================================
st.title("💊 의약품 처방량 예측 애플리케이션")
st.write("Prophet 모델을 사용하여 특정 의약품의 처방량을 예측하고 패턴을 분석합니다.")

st.sidebar.header("⚙️ 분석 설정")
csv_file = st.sidebar.file_uploader("진료 내역 데이터 (CSV)", type="csv")
xlsx_file = st.sidebar.file_uploader("의약품 정보 (XLSX)", type="xlsx")
target_code_input = st.sidebar.text_input("분석할 의약품 연합회코드 입력", "645902470")

# --- ✨ 조회 기간 선택 기능 추가 ✨ ---
st.sidebar.subheader("3. 그래프 조회 기간 설정")
# 사용자가 요청한 날짜를 기본값으로 설정
start_date = st.sidebar.date_input("조회 시작일", datetime.date(2024, 1, 1))
end_date = st.sidebar.date_input("조회 종료일", datetime.date(2024, 4, 30))

run_button = st.sidebar.button("🚀 예측 실행")


# ==============================================================================
# 📈 예측 및 시각화 실행
# ==============================================================================
if run_button:
    if csv_file and xlsx_file and target_code_input:
        if start_date > end_date:
            st.sidebar.error("오류: 조회 종료일은 시작일보다 빠를 수 없습니다.")
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
                         st.error(f"입력하신 코드 '{target_code}'가 데이터 파일의 컬럼에 존재하지 않습니다. 확인 후 다시 시도해주세요.")
                    else:
                        df_valid[target_code] = pd.to_numeric(df_valid[target_code], errors='coerce').fillna(0)
                        daily_sum = df_valid.groupby('일자')[target_code].sum()
                        daily_sum = daily_sum[daily_sum > 0]
                        
                        if daily_sum.empty:
                            st.error(f"입력하신 코드 '{target_code}'에 대한 처방 기록이 없습니다. 코드를 확인해주세요.")
                        else:
                            df_prophet = daily_sum.reset_index()
                            df_prophet.columns = ['ds', 'y']
                            
                            # --- ✨ 모델은 전체 데이터로 학습 ✨ ---
                            model = Prophet(daily_seasonality=True)
                            model.fit(df_prophet)

                            future = model.make_future_dataframe(periods=30, freq='D')
                            forecast = model.predict(future)

                            # --- ✨ 시각화를 위해 선택된 기간으로 데이터 필터링 ✨ ---
                            st.subheader(f"📊 종합 예측 결과 ({start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')})")

                            # 날짜 형식 통일
                            start_date_dt = pd.to_datetime(start_date)
                            end_date_dt = pd.to_datetime(end_date)
                            
                            # 그래프에 표시할 데이터만 필터링
                            plot_forecast = forecast[(forecast['ds'] >= start_date_dt) & (forecast['ds'] <= end_date_dt)]
                            plot_actual = df_prophet[(df_prophet['ds'] >= start_date_dt) & (df_prophet['ds'] <= end_date_dt)]

                            if plot_forecast.empty:
                                st.warning("선택하신 기간에 해당하는 예측 데이터가 없습니다. 기간을 다시 설정해주세요.")
                            else:
                                fig1, ax1 = plt.subplots(figsize=(14, 7))
                                
                                # 필터링된 데이터로 그래프 그리기
                                ax1.plot(plot_forecast['ds'], plot_forecast['yhat'], color='#0072B2', linestyle='-', linewidth=2, label='예측')
                                ax1.fill_between(plot_forecast['ds'], plot_forecast['yhat_lower'], plot_forecast['yhat_upper'], color='#0072B2', alpha=0.2)
                                ax1.plot(plot_actual['ds'], plot_actual['y'], 'k.', markersize=4, label='실제 처방량')

                                ax1.set_title(f"{drug_name} ({target_code}) 처방량 실제값 및 예측", fontsize=16)
                                ax1.set_xlabel("날짜", fontsize=12)
                                ax1.set_ylabel("처방 수량", fontsize=12)
                                ax1.legend()
                                ax1.grid(True, which='major', c='gray', ls='-', lw=1, alpha=0.2)
                                fig1.autofmt_xdate()
                                st.pyplot(fig1)

                            # --- 패턴 분석 그래프는 전체 기간으로 표시 ---
                            st.subheader("🔬 처방 패턴 상세 분석 (전체 기간)")
                            fig2 = model.plot_components(forecast)
                            st.pyplot(fig2)

                except Exception as e:
                    st.error(f"오류가 발생했습니다: {e}")
                    st.error("파일 인코딩(cp949)이나 내부 데이터 형식이 올바른지 확인해주세요.")
    else:
        st.warning("모든 파일을 업로드하고 약물 코드를 입력한 후 버튼을 눌러주세요.")
