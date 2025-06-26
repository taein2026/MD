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
# ✨ '오전/오후' 포함된 날짜/시간 처리용 함수 정의 ✨
# ==============================================================================
def parse_korean_datetime(s):
    try:
        s = str(s).replace('오전', 'AM').replace('오후', 'PM')
        return pd.to_datetime(s, format='%Y-%m-%d %p %I:%M:%S', errors='coerce')
    except:
        return pd.NaT

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
                    
                    target_code = target_code_input.strip()
                    drug_name = name_map.get(target_code, f"[{target_code}]")
                    st.success(f"분석 대상 의약품: **{drug_name}**")

                    if target_code not in df.columns or '진료일시' not in df.columns:
                         st.error(f"필수 컬럼('진료일시', '{target_code}')이 데이터 파일에 존재하지 않습니다.")
                    else:
                        # --- '오전/오후'를 포함한 '진료일시'를 ds 컬럼으로 변환 ---
                        df['ds'] = df['진료일시'].apply(parse_korean_datetime)
                        df.dropna(subset=['ds'], inplace=True)
                        
                        # --- Prophet 모델 학습을 위한 데이터 준비 ---
                        df_prophet_input = df[['ds']].copy()
                        df_prophet_input['y'] = pd.to_numeric(df[target_code], errors='coerce').fillna(0)
                        
                        start_date_dt = pd.to_datetime(train_start_date)
                        end_date_dt = pd.to_datetime(train_end_date).replace(hour=23, minute=59, second=59)
                        
                        # ✨ 학습용 데이터는 시간 정보가 포함된 원본 데이터를 사용
                        df_prophet_train = df_prophet_input[(df_prophet_input['ds'] >= start_date_dt) & (df_prophet_input['ds'] <= end_date_dt)]
                        
                        if df_prophet_train.empty:
                            st.error(f"선택하신 기간에 처방 기록이 없습니다.")
                        else:
                            # --- 모델 학습: 시간 정보가 포함된 데이터로 학습하여 패턴 파악 ---
                            model = Prophet(daily_seasonality=True)
                            model.fit(df_prophet_train)

                            # --- 예측: 일별(Daily) 기준으로 미래 예측 ---
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
                                col2.metric("재고 상태", "소진 예상", f"약 {days_left}일 후")
                                col3.metric("예상 소진일", f"{stock_out_date.strftime('%Y-%m-%d')}")
                                st.warning(f"**분석 요약:** 현재 재고({current_stock}개)는 약 {days_left}일 후 소진될 것으로 예측됩니다.")
                            else:
                                col2.metric("재고 상태", "재고 안정", "30일 내 소진 안됨")
                                thirty_days_later = end_date_dt + pd.Timedelta(days=30)
                                col3.metric("예상 소진일", f"{thirty_days_later.strftime('%Y-%m-%d')} 이후")
                                st.success(f"**분석 요약:** 현재 재고({current_stock}개)는 30일 내에 충분할 것으로 보입니다.")

                            # --- 종합 예측 그래프 시각화 ---
                            st.subheader(f"📊 {train_start_date.strftime('%Y-%m-%d')} ~ {train_end_date.strftime('%Y-%m-%d')} 데이터 학습 결과 및 30일 예측")
                            
                            # ✨ 실제값 표기를 위해 학습 데이터를 일별로 합산
                            actual_data_daily = df_prophet_train.set_index('ds').resample('D').sum().reset_index()

                            fig1, ax1 = plt.subplots(figsize=(14, 7))
                            
                            # Prophet의 기본 plot 기능을 사용하되, 실제값(검은 점)은 우리가 직접 그림
                            model.plot(forecast, ax=ax1)
                            ax1.plot(actual_data_daily['ds'], actual_data_daily['y'], 'k.', label='실제 처방량 (일별 총합)')
                            
                            ax1.axvline(x=end_date_dt, color='red', linestyle='--', linewidth=1.5, label='예측 시작일')
                            if not stock_out_day.empty:
                                ax1.axvline(x=stock_out_date, color='darkorange', linestyle=':', linewidth=2, label=f'재고 소진 예상일 ({days_left}일 후)')

                            ax1.set_title(f"{drug_name} ({target_code}) 처방량 예측", fontsize=16)
                            ax1.set_xlabel("날짜", fontsize=12)
                            ax1.set_ylabel("처방 수량", fontsize=12)
                            ax1.legend()
                            
                            st.pyplot(fig1)
                            
                            # --- 사용자 맞춤형 패턴 분석 그래프 ---
                            st.subheader("🔬 사용자 맞춤형 패턴 분석")
                            # 패턴 분석을 위해 시간 단위 예측 필요
                            analysis_future = model.make_future_dataframe(periods=1, freq='H') # 패턴만 볼 것이므로 기간은 짧게
                            analysis_forecast = model.predict(analysis_future)
                            
                            fig2, axes = plt.subplots(3, 1, figsize=(10, 15))
                            fig2.tight_layout(pad=5.0)

                            axes[0].plot(analysis_forecast['ds'], analysis_forecast['trend'], color='darkblue')
                            axes[0].set_title("장기적 처방량 추세", fontsize=14)
                            axes[0].set_xlabel("날짜")
                            axes[0].set_ylabel("처방량 변화")
                            axes[0].grid(True, linestyle='--', alpha=0.7)
                            
                            analysis_forecast['day_of_week'] = analysis_forecast['ds'].dt.day_name()
                            weekly_effect = analysis_forecast.groupby('day_of_week')['weekly'].mean()
                            day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
                            weekly_effect = weekly_effect.reindex(day_order)
                            kor_day_order = ["월", "화", "수", "목", "금", "토"]
                            weekly_effect.plot(kind='bar', ax=axes[1], color='skyblue', width=0.6, rot=0)
                            axes[1].set_title("주간 처방 패턴 (업무일 기준)", fontsize=14)
                            axes[1].set_xlabel("요일")
                            axes[1].set_ylabel("처방량 증감")
                            axes[1].grid(axis='y', linestyle='--', alpha=0.7)
                            axes[1].set_xticklabels(kor_day_order)
                            
                            axes[2].set_title("일간 처방 패턴 (업무 시간 기준)", fontsize=14)
                            single_day_data = analysis_forecast[analysis_forecast['ds'].dt.date == analysis_forecast['ds'].dt.date.min()].copy()
                            axes[2].plot(single_day_data['ds'], single_day_data['daily'], color='lightgreen', linewidth=2)
                            axes[2].grid(linestyle='--', alpha=0.7)
                            axes[2].set_xlabel("시간")
                            axes[2].set_ylabel("처방량 증감")
                            axes[2].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                            start_time = pd.to_datetime(single_day_data['ds'].dt.date.iloc[0]) + pd.DateOffset(hours=8)
                            end_time = pd.to_datetime(single_day_data['ds'].dt.date.iloc[0]) + pd.DateOffset(hours=19)
                            axes[2].set_xlim([start_time, end_time])
                            
                            st.pyplot(fig2)

                except Exception as e:
                    st.error(f"분석 중 오류가 발생했습니다: {e}")
    else:
        st.warning("모든 파일을 업로드하고 약물 코드를 입력한 후 버튼을 눌러주세요.")
