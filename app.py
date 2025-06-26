# app.py

# ✅ 1. 라이브러리 불러오기
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from prophet import Prophet
import io

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
run_button = st.sidebar.button("🚀 예측 실행")


# ==============================================================================
# 📈 예측 및 시각화 실행
# ==============================================================================
if run_button:
    if csv_file and xlsx_file and target_code_input:
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
                        # ✅ 8. Prophet 입력용 데이터프레임 구성 및 데이터 그룹 나누기 (✨수정된 부분✨)
                        df_prophet = daily_sum.reset_index()
                        df_prophet.columns = ['ds', 'y']
                        
                        # 데이터 포인트 간의 시간 차이가 30일 이상이면 새로운 그룹으로 간주
                        df_prophet['group'] = (df_prophet['ds'].diff() > pd.Timedelta('30 days')).cumsum()

                        # ✅ 9. Prophet 모델 학습
                        model = Prophet(daily_seasonality=True)
                        model.fit(df_prophet)

                        # ✅ 10. 향후 예측
                        future = model.make_future_dataframe(periods=30, freq='D') # 미래 예측은 매일
                        forecast = model.predict(future)

                        # ✅ 11. 결과 시각화 (✨그래프 그리는 방식 수정✨)
                        st.subheader("📊 종합 예측 결과")
                        
                        # 과거 데이터와 미래 예측 분리
                        last_date = df_prophet['ds'].max()
                        history_fc = forecast[forecast['ds'] <= last_date].copy()
                        
                        # 과거 데이터에 그룹 정보 추가
                        history_fc['group'] = pd.merge(history_fc, df_prophet, on='ds')['group']

                        future_fc = forecast[forecast['ds'] > last_date]
                        
                        # 그래프 객체 생성
                        fig1, ax1 = plt.subplots(figsize=(14, 7))

                        # --- 그룹별로 과거 데이터 적합선을 끊어서 그리기 ---
                        for i, group in history_fc.groupby('group'):
                            # 첫 번째 그룹에만 레이블을 추가하여 범례가 중복되지 않도록 함
                            label = '과거 데이터 모델 적합' if i == 0 else None
                            ax1.plot(group['ds'], group['yhat'], color='gray', linestyle='-', linewidth=1.5, label=label)

                        # 미래 예측 기간의 예측선과 불확실성 구간 그리기
                        ax1.plot(future_fc['ds'], future_fc['yhat'], color='#0072B2', linestyle='-', linewidth=2, label='미래 예측')
                        ax1.fill_between(future_fc['ds'], future_fc['yhat_lower'], future_fc['yhat_upper'], color='#0072B2', alpha=0.2)

                        # 실제 데이터 점(검은색) 그리기
                        ax1.plot(df_prophet['ds'], df_prophet['y'], 'k.', markersize=4, label='실제 처방량')

                        # 그래프 꾸미기 (제목, 라벨, 예측 시작선 등)
                        ax1.set_title(f"{drug_name} ({target_code}) 처방량 실제값 및 예측", fontsize=16)
                        ax1.set_xlabel("날짜", fontsize=12)
                        ax1.set_ylabel("처방 수량", fontsize=12)
                        ax1.axvline(x=last_date, color='red', linestyle='--', linewidth=1.5, label='예측 시작일')
                        ax1.legend()
                        ax1.grid(True, which='major', c='gray', ls='-', lw=1, alpha=0.2)
                        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
                        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                        fig1.autofmt_xdate()
                        st.pyplot(fig1)

                        # --- ✅ 12. 구성 요소 분해 시각화 ---
                        st.subheader("🔬 처방 패턴 상세 분석")
                        fig2 = model.plot_components(forecast)
                        # ... (이하 코드는 동일) ...
                        st.pyplot(fig2)

            except Exception as e:
                st.error(f"오류가 발생했습니다: {e}")
                st.error("파일 인코딩(cp949)이나 내부 데이터 형식이 올바른지 확인해주세요.")
    else:
        st.warning("모든 파일을 업로드하고 약물 코드를 입력한 후 버튼을 눌러주세요.")
