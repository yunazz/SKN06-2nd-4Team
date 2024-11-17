from preprocessing import DataPreprocessor
from dataloader import load_dataset
import joblib
import numpy as np
import pandas as pd
import sys
import os

import streamlit as st
st.set_page_config(layout="wide")

@st.cache_resource 
def load_model_tree():
    return joblib.load('models/best_tree.pkl')

@st.cache_resource 
def load_model_rf():
    return joblib.load('models/best_rf.pkl')

@st.cache_resource 
def load_model_gb():
    return joblib.load('models/best_gb.pkl')

@st.cache_resource 
def load_model_xgb():
    return joblib.load('models/best_xgb.pkl')

# 모델 로드
model_tree = load_model_tree()
model_rf = load_model_rf()
model_gb = load_model_gb()
model_xgb = load_model_xgb()

# 전처리기 로드
preprocessor = DataPreprocessor()


# Front-End 코드
# 스타일 css
st.markdown(
    """
    <style>
    .custom-button {background-color: #4CAF50;color: white;padding: 10px 20px;font-size: 16px;border-radius: 5px;border: none;cursor: pointer;}
    .custom-button:hover {background-color: #45a049;}
    .stButton {display: flex;justify-content: center;}
    .stButton button {background-color: #55c9c2;color: white!important;padding: 12px 80px;font-size: 24px;border-radius: 5px;border: none;cursor: pointer;transition: all 0.2s ease;}
    .stButton button:hover {background-color: #4db5ae!important;color: white;}
    p.final_prediction {font-size: 42px;text-align: center;font-weight: 800;margin-top: 120px;}
    p.final_prediction.positive::before,p.final_prediction.negative::before {content: "";position: absolute;top: 80px; left: 50%;border-top: 12px solid #ddd; transform: translateX(-50%);text-align: center;width: 100px;    }
    p.final_prediction.positive::after,p.final_prediction.negative::after {display: block;text-align:center;font-size: 40px;line-height: 50px;}
    p.final_prediction.positive::after {content: "고객 유지";}
    p.final_prediction.negative::after {content: "고객 이탈";}
    #root > div:nth-child(1) > div.withScreencast > div > div > section > div.stMainBlockContainer.block-container.st-emotion-cache-1jicfl2.ea3mdgi5 > div > div > div > div:nth-child(19) > div > div > p {text-align: center;font-size: 20px;}
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<h1 style="text-align: center; margin-bottom: 0">신용카드 이용 고객 이탈 예측</h1>', unsafe_allow_html=True)
st.markdown('<hr style="margin: 16px 0 36px; border-color: #ccc; border:2px solid #ddd"/>', unsafe_allow_html=True)

# 고객 정보 입력
st.subheader('📌 고객 정보 입력')

col1, col2, col3, col4 = st.columns(4)
with col1:
    age = st.selectbox('**고객 나이**', range(18,110), index=12)
    education_level = st.selectbox('**학력**', [("학력 없음", "Uneducated"), ("고등학교 재학/졸업", "High School" ), ("전문학사 재학/졸업", "College", ), ("학사 재학/졸업", "Graduate"), ("석사 재학/졸업", "Post-Graduate"), ("박사 재학/졸업", "Doctorate")], format_func=lambda x: x[0], index=3)
with col2:
    gender = st.selectbox("**성별**", ['M', 'F'], index=1)
    income_category = st.selectbox('**소득 수준**', ['Less than $40K', '$40K - $60K', '$80K - $120K', '$60K - $80K', '$120K +'], index=2)
with col3:
    dependent_cnt = st.selectbox("**부양가족수**", range(0,7), index=0)
with col4:
    marital_status = st.selectbox('**결혼 여부**', [ ("미혼",'Single'), ("기혼", 'Married'), ("이혼",'Divorced')], format_func=lambda x: x[0], index=0)

st.markdown('<hr style="margin: 16px 0;"/>', unsafe_allow_html=True)

col5, col6, col7, col8 = st.columns(4)

with col5:
    card_category = st.selectbox("**카드 종류**", ['Blue', 'Silver', 'Gold', 'Platinum'], index=0)
    visit_cnt_in_year = st.selectbox('**연간 은행 방문 수**', [("1회 미만",0), ("1 ~ 10회",1), ("11 ~ 20회",2), ("21 ~ 30회",3), ("31 ~ 40회",4), ("41회 이상",5)], format_func=lambda x: x[0], index=1)
with col6:
    card_usage_period = st.slider('**카드 사용 기간(개월)**', min_value=1, max_value=60, value=12)
with col7:
    account_cnt = st.selectbox('**계좌 수**', range(1,7), index=0)
with col8:
    inactive_month_in_year = st.selectbox('**연내 계좌 비활성 기간**',  [("15일 미만",0), ("15일 이상 ~ 1개월 미만",1), ("1개월 이상 ~ 2개월 미만",2), ("2개월 ~ 4개월 미만",3), ("4개월 이상 ~ 6개월 미만",4), ("6개월 이상",5)], format_func=lambda x: x[0], index=0)

st.markdown('<hr style="margin: 16px 0;"/>', unsafe_allow_html=True)

col9, col10, col11, col12 = st.columns(4)
with col9:
    avg_remain_credit_limit = st.slider('**평균 잔여 신용 한도**', min_value=1500, max_value=35000, value=20000)
    total_cnt_change_q4_q1 = st.slider('**총 거래 횟수 변화율(4분기/1분기)**', min_value=0.0, max_value=3.0, value=1.5)
with col10:
    total_amt_change_q4_q1 = st.slider('**연간 거래액 변화율(4분기/1분기)**',  min_value=0.0, max_value=3.0, value=1.5)
    avg_utilization_ratio = st.slider('**카드 한도 대비 잔액의 비율**', min_value=0.0, max_value=1.0, value=0.5)
with col11:
    credit_limit = st.slider('**신용 한도**', min_value=1500, max_value=35000, value=20000)
    total_trans_amt = st.slider('**총 거래 금액**',  min_value=8000, max_value=35000, value=20000)
with col12:
    revolving_balance = st.slider('**잔금**', min_value=1500, max_value=35000, value=10000)
    total_trans_cnt = st.slider('**총 거래 횟수 ⭐**', min_value=10, max_value=100, value=60)


# 고객 예측
st.divider()
st.markdown('<h3 style="margin-bottom: 0; transform: translateY(10px)">⚙️ 머신 러닝 모델</h2>', unsafe_allow_html=True)

model_filter = st.selectbox("", ['XGBoost', 'Gradient Boosting', 'Random Forest', 'Decision Tree'], index=0)

# 예측 버튼
input_data = {
    'age': age,
    'gender': gender,
    'dependent_cnt': dependent_cnt,
    'education_level': education_level[1],
    'income_category': income_category,
    'card_usage_period': card_usage_period,
    'account_cnt': account_cnt,
    'inactive_month_in_year': inactive_month_in_year[1],
    'visit_cnt_in_year': visit_cnt_in_year[1],
    'credit_limit': credit_limit,
    'revolving_balance': revolving_balance,
    'avg_remain_credit_limit': avg_remain_credit_limit,
    'total_amt_change_q4_q1': total_amt_change_q4_q1,
    'total_trans_amt': total_trans_amt,
    'total_trans_cnt': total_trans_cnt,
    'total_cnt_change_q4_q1': total_cnt_change_q4_q1,
    'avg_utilization_ratio': avg_utilization_ratio,
    'marital_status': marital_status[1],
    'card_category': card_category
}
    
if st.button('예측하기'):
    input_data_df = pd.DataFrame([input_data])

    processed_data = preprocessor.preprocess(input_data_df)
    
    prediction = None
    prediction_proba = None
    
    if model_filter == 'Decision Tree':
        prediction = model_tree.predict(processed_data)
        prediction_proba = model_tree.predict_proba(processed_data)[:, 1]

    elif model_filter == 'Random Forest':
        prediction = model_rf.predict(processed_data)
        prediction_proba = model_rf.predict_proba(processed_data)[:, 1]

    elif model_filter == 'Gradient Boosting':
        prediction = model_gb.predict(processed_data)
        prediction_proba = model_gb.predict_proba(processed_data)[:, 1]

    elif model_filter == 'XGBoost':
        prediction = model_xgb.predict(processed_data)
        prediction_proba = model_xgb.predict_proba(processed_data)[:, 1]
    
    # 임계값 설정
    thresh = 0.9
    final_prediction = np.where(prediction_proba >= thresh, 1, 0)
    final_prediction_txt = "고객 이탈" if final_prediction == 1 else "고객 유지"
    
    if prediction is not None and prediction_proba is not None:
        # st.markdown('<h2 class="final_prediction" style="text-align: center; margin: 50px 0 0">예측 결과</h2>', unsafe_allow_html=True)
        if final_prediction == 1: 
            st.markdown('<p class="final_prediction negative" style="font-size: 70px">👋👋👋</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="final_prediction positive" style="font-size: 70px">👏👏👏</p>', unsafe_allow_html=True)
        
        st.write(f'이탈할 확률: {prediction_proba[0]*100:.2f}%')
    else:
        st.error("예측을 수행할 수 없습니다. 모델을 확인해주세요.")