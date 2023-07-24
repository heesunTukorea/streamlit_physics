from matplotlib import pyplot as plt
import streamlit as st
import pandas as pd
import numpy as np
from cache import *
import os
import datetime
from sklearn.model_selection import train_test_split
import sklearn
import seaborn as sns
from sklearn import tree
from sklearn.metrics import mean_squared_error, accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, roc_curve, classification_report

# 화면 분할
col1, col2 = st.columns([7, 2])

with col2:
    st.image("tuk_img.png")

st.title('모델학습과 성능 측정')

with st.sidebar:
    selecop = st.selectbox('카테고리', ('데이터 보기', 'Logistic'))

df_QC = add_QC(dedicated_data)
sensor_data_df = control_sensor_data(df_QC)
new_df = calculator_data(df_list, df_list2)
new_df_shape = new_df.shape

# 데이터 보기 선택
if selecop == '데이터 보기':
    st.header("로트별 데이터")
    st.dataframe(new_df)
    st.write(new_df.shape)
    if st.button("상관관계 보기"):
        correlation1 = correlation_def1(new_df)
        if correlation1 is not None: 
            st.title("상관관계 분석")
            st.write(correlation1)
            fig1, ax1 = plt.subplots(figsize=(10, 8))
            sns.heatmap(correlation1, annot=True, fmt=".2f", ax=ax1)
            st.pyplot(fig1)



if selecop == "Logistic":
    st.title("로지스틱 모델")
    col11, col12, col13 = st.columns([2, 2, 2])
    with col11:
        class_weight1 = st.number_input("class_weight", value=0.018, step=0.01)
    with col12:
        split_size = st.number_input("test_size", value=0.3, step=0.1)
    with col13:
        threshold1 = st.number_input("Threshold", value=0.5, step=0.1)
    

    if st.button('설정 완료'):
        y_test2,y_pred_adjusted,class_ratio, accuracy, confusion_matrix_df, fig_lo, lo_model = rogistic_model(new_df, df_QC, class_weight1,split_size, threshold1)
        st.write("추천 weight:", class_ratio)
        if lo_model is not None:
            st.success("로지스틱 모델 학습이 완료되었습니다!")
            st.write("모델 정보:")
            st.write(lo_model)
            st.pyplot(fig_lo)

        else:
            st.warning("로지스틱 모델 학습에 실패했습니다.")

        st.write("로지스틱 모델 평가")
        st.write("Accuracy:", accuracy)
        st.write("Confusion Matrix:")
        st.dataframe(confusion_matrix_df)

        # df_list = split_dataframe_by_lot(df_QC)
        # new_error_df = lot_data2(df_list)

        # st.subheader("에러 데이터")
        # st.dataframe(new_error_df)
            
        # st.subheader("오류값")
        # df_list=split_dataframe_by_lot(y_pred_adjusted,split_size,sensor_data_df)
        # new_error_df = lot_data2(df_list)
        # st.dataframe(new_error_df)

with st.sidebar:
    st.sidebar.markdown("## 목록")
    st.write("총괄리더: 신형찬")
    st.write("분석팀장: 안이찬")
    st.write("개발팀장: 양희선")
    st.write("팀원: 정원석")