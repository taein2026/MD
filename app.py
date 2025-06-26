# app.py

# ✅ 1. 라이브러리 불러오기
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from prophet import Prophet
import io

# ==============================================================================
# ✨ Matplotlib 한글 폰트 설정 (가장 먼저 실행) ✨
# ==============================================================================
try:
    plt.rc('font', family='Malgun Gothic') # Windows
except:
    try:
        plt.rc('font', family='AppleGothic') # Mac
    except:
        # 로컬에서 실행 시 아래 경로에 폰트 파일이 있어야 합니다.
        try:
            plt.rc('font', family='NanumGothic')
        except:
            pass # 폰트가 없어도 앱은 실행되도록 함

plt.rcParams['axes.unicode_minus'] = False # 마이너스 기호 깨짐 방지


# ==============================================================================
# 💻 웹 애플리케이션 UI 구성
# ==============================================================================
st.title("💊 의약품 처방량 예측 애플리케이션")
st.write("Prophet 모델을 사용하여 특정 의약품의 처방량을 예측하고 패턴을 분석합니다.")

# --- 사이드바 ---
st.sidebar.header("⚙️ 분석 설정")

# 1. 파일 업로드
st.sidebar.subheader("1. 데이터 파일 업로드")
csv_file = st.sidebar.file_uploader("진료 내역 데이터 (CSV)", type="csv")
xlsx_file = st.sidebar.file_uploader("의약품 정보 (XLSX)", type="xlsx")

# 2. 약물 코드 입력
st.sidebar.subheader("2. 분석 대상 입력")
target_code_input = st.sidebar.text_input("분석할 의약품 연합회코드 입력", "645902470")

# 3. 예측 실행 버튼
run_button = st.sidebar.button("🚀 예측 실행")


# ==============================================================================
# 📈 예측 및 시각화 실행
# ==============================================================================
if run_button:
    if csv_file and xlsx_file and target_code_input:
        with st.spinner('데이터를 처리하고 모델을 학습하는 중입니다... 잠시만 기다려 주세요.'):
            try:
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
                        
                        model = Prophet(daily_seasonality=True)
                        model.fit(df_prophet)

                        # --- 향후 예측 ---
                        past_dates = df_prophet[['ds']]
                        future_weekdays = pd.bdate_range(start=df_prophet['ds'].max(), periods=31)[1:]
                        future_dates = pd.DataFrame({'ds': future_weekdays})
                        total_dates = pd.concat([past_dates, future_dates])
                        forecast = model.predict(total_dates)

                        # --- 결과 시각화 1 ---
                        st.subheader("📊 종합 예측 결과")
                        last_date = df_prophet['ds'].max()
                        history_fc = forecast[forecast['ds'] <= last_date]
                        future_fc = forecast[forecast['ds'] > last_date]
                        fig1, ax1 = plt.subplots(figsize=(14, 7))
                        ax1.plot(history_fc['ds'], history_fc['yhat'], color='gray', linestyle='-', linewidth=1.5, label='과거 데이터 모델 적합')
                        ax1.plot(future_fc['ds'], future_fc['yhat'], color='#0072B2', linestyle='-', linewidth=2, label='미래 예측')
                        ax1.fill_between(future_fc['ds'], future_fc['yhat_lower'], future_fc['yhat_upper'], color='#0072B2', alpha=0.2)
                        ax1.plot(df_prophet['ds'], df_prophet['y'], 'k.', markersize=4, label='실제 처방량')
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

                        # --- 결과 시각화 2 ---
                        st.subheader("🔬 처방 패턴 상세 분석")
                        fig2 = model.plot_components(forecast)
                        title_map = {'trend': '장기적 추세','weekly': '주간 패턴','yearly': '연간 패턴','daily': '일간 패턴'}
                        weekday_map_kor = ['일요일', '월요일', '화요일', '수요일', '목요일', '금요일', '토요일']
                        
                        for ax in fig2.get_axes():
                            current_title = ax.get_title()
                            if current_title in title_map:
                                ax.set_title(title_map[current_title], fontsize=14)
                            if current_title == 'weekly':
                                ax.set_xticks(range(7))
                                ax.set_xticklabels(weekday_map_kor)
                                ax.set_xlabel('요일', fontsize=12)
                            if current_title == 'yearly':
                                ax.set_xlabel('연중 날짜', fontsize=12)
                            if current_title == 'daily':
                                ax.set_xlabel('하루 중 시간', fontsize=12)
                        fig2.tight_layout()
                        st.pyplot(fig2)

            except Exception as e:
                st.error(f"오류가 발생했습니다: {e}")
                st.error("파일 인코딩(cp949)이나 내부 데이터 형식이 올바른지 확인해주세요.")
    else:
        st.warning("모든 파일을 업로드하고 약물 코드를 입력한 후 버튼을 눌러주세요.")