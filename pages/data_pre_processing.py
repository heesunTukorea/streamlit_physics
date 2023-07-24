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
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D


#페이지 할당
st.set_page_config(page_title="data_fre_processing")

#화면 분할
col1, col2 = st.columns([7, 2])

with col2:
    st.image("tuk_img.png")

st.title('데이터 분석')
#사이드바의 선택 박스 
with st.sidebar:
    selecop = st.selectbox('데이터 선택', ('병합_csv', '오류값 생성_csv','로트별 데이터',"PCA"))

    




fig = None
fig1 = None
#병합 데이터 프레임과 간단 분석
if selecop == '병합_csv':
    st.header("데이터 병합")
    if st.button('데이터프레임 생성'):
        st.write("모든 csv파일이 합쳐진 데이터")
        df_m = load_and_merge_csvs(file_list)
        st.dataframe(df_m)
        df_m_shape = df_m.shape
        st.write(" 데이터 형태:", df_m_shape)

        
        #selecop1 = st.selectbox('데이터 간단 분석', ('','상관관계 분석', '데이터 분포'))
        #if selecop1 == '상관관계 분석':

        correlation = correlation_def(df_m)

        st.title("상관관계 분석")
        st.write(correlation)
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(correlation, annot=True, fmt=".2f", ax=ax)
        st.pyplot(fig)
    

        #elif selecop1 == '데이터 분포':

        st.title("데이터 분포")
        # Create the histogram plot
        fig, ax = plt.subplots(figsize=(10, 10))
        df_m.hist(ax=ax)
        plt.tight_layout()
        st.pyplot(fig)
        

df_QC = add_QC(dedicated_data)
#오류값을 나눈 데이터 프레임과 간단 분석
if selecop == '오류값 생성_csv':
    st.header("오류값 생성")
    if st.button('데이터프레임 생성'):
        st.write("QC:오류값이 생성된 데이터")
        st.dataframe(df_QC)
        df_QC_shape = df_QC.shape
        st.write(" 데이터 형태:", df_QC_shape)
        st.write("QC 갯수", df_QC['QC'].value_counts())
        
        df_QC1 = df_QC[['Lot', 'pH', 'Temp', 'Current', 'Voltage', 'QC']]
        #selecop2 = st.selectbox('데이터 간단 분석', ('','상관관계 분석', '데이터 분포'))
        #if selecop2 == '상관관계 분석':
        correlation1 = correlation_def(df_QC)
        st.title("상관관계 분석")
        st.write(correlation1)
        fig1, ax1 = plt.subplots(figsize=(10, 8))
        sns.heatmap(correlation1, annot=True, fmt=".2f", ax=ax1)
        st.pyplot(fig1)


        #elif selecop2 == '데이터 분포':
            
        st.title("데이터 분포")
        # Create the histogram plot
        
        fig1, ax1 = plt.subplots(figsize=(10, 10))
        df_QC1.hist(ax=ax1)
        plt.tight_layout()
        st.pyplot(fig1)
        
#로트별 데이터로 나누고 범위를 조정 시각화 하는 코드            
if selecop == '로트별 데이터':
    st.header("로트별 데이터")
    sensors_of_lots(X_data)
    df_list, df_list2, columns = sensors_of_lots(X_data)
    len_df_list = len(df_list)
    len_df_list2 = len(df_list2)

    normal_lot_range_placeholder = st.empty()
    unnormal_lot_range_placeholder = st.empty()

    col3, col4 = st.columns([5, 5])

    
    with col3:
        normal_lot_start = int(st.number_input('범위의 최솟값을 정해주세요'))
        st.write('선택한 정상 로트의 범위 min ', normal_lot_start)
        st.write('선택범위: sensor{} - {}'.format(0, len_df_list))
    with col4:
        normal_lot_finish = int(st.number_input('범위의 최댓값을 정해주세요'))
        st.write('선택한 정상 로트의 범위 max  ', normal_lot_finish)
        st.write('불량 로트 선택범위: sensor{} - {}'.format(0, len_df_list2))

    
    unnormal_lot_start, unnormal_lot_finish = st.slider('불량 로트 범위', 0, len_df_list2, (0, 9))

    if st.button('시각화 그림 생성'):
        fig = plot_in_sensors(X_data,df_list, df_list2, normal_lot_start, normal_lot_finish, unnormal_lot_start,
                            unnormal_lot_finish, columns)

        st.title("로트별 데이터")
        st.pyplot(fig)

#PCA
# PCA
# PCA
if selecop == 'PCA':
    
    st.title("PCA")
    
    df_QC1 = df_QC[["pH", "Current", "Voltage", "Temp", "QC"]]
    df_pc = df_QC[["pH", "Current", "Voltage", "Temp"]]
    st.subheader("데이터")
    st.dataframe(df_QC1)
    st.write("영향 변수: pH,Current,Voltage,Temp")
    # PCA 변환된 데이터 얻기
    df_pca, components = pca_sensor_data(df_QC1, df_pc)

    st.title("2D PCA")
    fig = plt.figure(figsize=(8, 8))
    plt.scatter(df_pca.loc[df_pca['QC'] == 1, 'PC1'], df_pca.loc[df_pca['QC'] == 1, 'PC2'], c='red', alpha=0.3, label='Nomal')
    plt.scatter(df_pca.loc[df_pca['QC'] == 0, 'PC1'], df_pca.loc[df_pca['QC'] == 0, 'PC2'], c='blue', label='Anomaly')
    plt.legend()

    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('PCA')

    feature_names = df_pc.columns
    for i, feature in enumerate(feature_names):
        plt.arrow(0, 0, components[0, i], components[1, i], color='r', alpha=0.5)
        plt.text(components[0, i] * 1.1, components[1, i] * 1.1, feature, color='r')

    st.pyplot(fig)

    if st.button("3D PCA"):
        st.title("3D PCA")
        df_pca_3D, components_3D = pca_sensor_data(df_QC1, df_pc)  # 2개의 주성분으로 PCA 실행
        fig = plot_pca_3D(df_pca_3D)
        st.pyplot(fig)



with st.sidebar:
    st.sidebar.markdown("## 목록")
    st.write("총괄리더: 신형찬")
    st.write("분석팀장: 안이찬")
    st.write("개발팀장: 양희선")
    st.write("팀원: 정원석")
