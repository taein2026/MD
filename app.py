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
st.title("💊 의약품 30일 재고 예측 시스템")
st.write("지정한 기간의 데이터로 학습하여, **향후 30일간의 재고 상태**를 분석합니다.")

st.sidebar.header("⚙️ 분석 설정")
csv_file = st.sidebar.file_uploader("진료 내역 데이터 (CSV)", type="csv")
xlsx_file = st.sidebar.file_uploader("의약품 정보 (XLSX)", type="xlsx")
target_code_input = st.sidebar.text_input("분석할 의약품 연합회코드 입력", "645902470")

st.sidebar.subheader("3. 예측 시나리오 설정")
train_start_date = st.sidebar.date_input("학습 시작일", datetime.date(2023, 1, 1))
train_end_date = st.sidebar.date_input("학습 종료일 (이 날짜를 기준으로 예측)", datetime.date(2023, 12, 31))

st.sidebar.subheader("4. 재고 분석 설정")
current_stock = st.sidebar.number_input("현재 재고량 입력", min_value=0, value=100)
forecast_period = 30 

run_button = st.sidebar.button("🚀 분석 실행")

# ==============================================================================
# 📈 예측 및 시각화 실행
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
                            
                            # --- 재고 소진일 계산 ---
                            future_fc_stock = forecast[forecast['ds'] > end_date_dt].copy()
                            future_fc_stock['yhat'] = future_fc_stock['yhat'].clip(lower=0)
                            future_fc_stock['cumulative_yhat'] = future_fc_stock['yhat'].cumsum()
                            stock_out_day = future_fc_stock[future_fc_stock['cumulative_yhat'] >= current_stock]

                            # --- 결과 텍스트 출력 ---
                            st.subheader("📦 30일 재고 분석 결과")
                            col1, col2, col3 = st.columns(3)
                            col1.metric("현재 재고량", f"{current_stock} 개")

                            if not stock_out_day.empty:
                                stock_out_date = stock_out_day.iloc[0]['ds']
                                days_left = (stock_out_date - end_date_dt).days
                                col2.metric("재고 상태", "소진 예상", f"-{days_left}일 후 소진")
                                col3.metric("예상 소진일", f"{stock_out_date.strftime('%Y-%m-%d')}")
                                st.warning(f"**분석 요약:** 현재 재고({current_stock}개)는 앞으로 **약 {days_left}일** 후인 **{stock_out_date.strftime('%Y-%m-%d')}** 경에 소진될 것으로 예측됩니다. 재고 보충이 필요합니다.")
                            else:
                                col2.metric("재고 상태", "재고 안정", "30일 내 소진 안됨")
                                thirty_days_later = end_date_dt + pd.Timedelta(days=30)
                                col3.metric("예상 소진일", f"{thirty_days_later.strftime('%Y-%m-%d')} 이후")
                                st.success(f"**분석 요약:** 현재 재고({current_stock}개)는 예측 기간인 **30일** 내에는 충분할 것으로 보입니다.")

                            # --- 종합 예측 그래프 시각화 ---
                            st.subheader(f"📊 {train_start_date.strftime('%Y-%m-%d')} ~ {train_end_date.strftime('%Y-%m-%d')} 데이터 학습 결과 및 30일 예측")
                            # ... (이전과 동일하여 코드 생략) ...

                            # --- ✨ 사용자 맞춤형 패턴 분석 그래프 (✨수정된 부분✨) ---
                            st.subheader("🔬 사용자 맞춤형 패턴 분석")
                            
                            # 1. 트렌드(Trend) 그래프
                            fig_trend, ax_trend = plt.subplots(figsize=(10, 4))
                            model.plot_trend(forecast, ax=ax_trend)
                            ax_trend.set_title("장기적 처방량 추세")
                            ax_trend.set_xlabel("날짜")
                            ax_trend.set_ylabel("처방량 변화")
                            st.pyplot(fig_trend)
                            
                            # 2. 주간 패턴(Weekly) - 막대그래프로 업무일만 표시
                            # 요일별 평균 'weekly' 효과 계산
                            forecast['day_of_week'] = forecast['ds'].dt.day_name()
                            weekly_effect = forecast.groupby('day_of_week')['weekly'].mean()
                            # 월요일부터 토요일 순서로 정렬
                            day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
                            weekly_effect = weekly_effect.reindex(day_order)
                            kor_day_order = ["월", "화", "수", "목", "금", "토"]

                            fig_weekly, ax_weekly = plt.subplots(figsize=(10, 4))
                            weekly_effect.plot(kind='bar', ax=ax_weekly, color='skyblue', width=0.6, rot=0)
                            ax_weekly.set_title("주간 처방 패턴 (업무일 기준)")
                            ax_weekly.set_xticklabels(kor_day_order)
                            ax_weekly.set_xlabel("요일")
                            ax_weekly.set_ylabel("처방량 증감")
                            ax_weekly.grid(axis='y', linestyle='--', alpha=0.7)
                            st.pyplot(fig_weekly)

                            # 3. 일간 패턴(Daily) - 업무 시간(9-18시)만 표시
                            # 하루 중 시간대별 'daily' 효과 계산
                            forecast['time'] = forecast['ds'].dt.time
                            daily_effect = forecast.groupby('time')['daily'].mean()
                            
                            fig_daily, ax_daily = plt.subplots(figsize=(10, 4))
                            daily_effect.plot(ax=ax_daily, color='lightgreen')
                            ax_daily.set_title("일간 처방 패턴 (업무 시간 기준)")
                            # x축을 8시부터 19시까지로 제한
                            ax_daily.set_xlim([datetime.time(8, 0), datetime.time(19, 0)])
                            ax_daily.set_xlabel("시간")
                            ax_daily.set_ylabel("처방량 증감")
                            ax_daily.grid(linestyle='--', alpha=0.7)
                            st.pyplot(fig_daily)

                except Exception as e:
                    st.error(f"오류가 발생했습니다: {e}")
    else:
        st.warning("모든 파일을 업로드하고 약물 코드를 입력한 후 버튼을 눌러주세요.")
