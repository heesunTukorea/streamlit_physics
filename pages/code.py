import streamlit as st



col1, col2 = st.columns([7,2])
st.title("코드")
with col2:
    st.image("tuk_img.png")
with st.sidebar:
    selecop = st.selectbox('코드 선택', ('data_pre_processing.py', 'data_set.py','cache.py'))
if selecop == 'data_pre_processing.py':
    code = '''
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

    #페이지 할당
    st.set_page_config(page_title="data_fre_processing")

    #화면 분할
    col1, col2 = st.columns([7, 2])

    with col2:
        st.image("tuk_img.png")

    st.title('데이터 병합')
    #사이드바의 선택 박스 
    with st.sidebar:
        selecop = st.selectbox('데이터 선택', ('병합_csv', '오류값 생성_csv','로트별 데이터'))
    if selecop == '병합_csv' or selecop == '오류값 생성_csv':
        with st.sidebar:
            selecop1 = st.selectbox('데이터 간단 분석', ('상관관계 분석', '데이터 분포'))

        



    fig = None
    fig1 = None
    #병합 데이터 프레임과 간단 분석
    if selecop == '병합_csv':
        if st.button('데이터프레임 생성'):
            df_m = load_and_merge_csvs(file_list)
            st.dataframe(df_m)
            df_m_shape = df_m.shape
            st.write(" 데이터 형태:", df_m_shape)

            if selecop1 == '상관관계 분석':
                if st.button('상관관계 분석 생성'):
                    correlation = correlation_def(df_m)
                    if correlation is not None:
                        st.title("상관관계 분석")
                        st.write(correlation)
                        fig, ax = plt.subplots(figsize=(10, 8))
                        sns.heatmap(correlation, annot=True, fmt=".2f", ax=ax)
                        st.pyplot(fig)

            if selecop1 == '데이터 분포':
                if st.button('상관관계 분석 생성'):
                    st.title("데이터 분포")
                    # Create the histogram plot
                    fig, ax = plt.subplots(figsize=(10, 10))
                    df_m.hist(ax=ax)
                    plt.tight_layout()


    #오류값을 나눈 데이터 프레임과 간단 분석
    if selecop == '오류값 생성_csv':
        if st.button('데이터프레임 생성'):
            df_QC = add_QC(dedicated_data)
            st.dataframe(df_QC)
            df_QC_shape = df_QC.shape
            st.write(" 데이터 형태:", df_QC_shape)
            st.write("QC 갯수", df_QC['QC'].value_counts())
            if selecop1 == '상관관계 분석':
                if st.button('상관관계 분석 생성'):
                    correlation1 = correlation_def(df_QC)
                    if correlation1 is not None: 
                        st.title("상관관계 분석")
                        st.write(correlation1)
                        fig1, ax1 = plt.subplots(figsize=(10, 8))
                        sns.heatmap(correlation1, annot=True, fmt=".2f", ax=ax1)
                        st.pyplot(fig1)

            if selecop1 == '데이터 분포':
                if st.button('데이터분포 분석 생성'):
                    st.title("데이터 분포")
                    # Create the histogram plot
                    df_QC1 = df_QC[['Lot', 'pH', 'Temp', 'Current', 'Voltage', 'QC']]

                    fig1, ax1 = plt.subplots(figsize=(10, 10))
                    df_QC1.hist(ax=ax1)
                    plt.tight_layout()
                    st.pyplot(fig1)

    #로트별 데이터로 나누고 범위를 조정 시각화 하는 코드            
    if selecop == '로트별 데이터':

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
            fig = plot_in_sensors(df_list, df_list2, normal_lot_start, normal_lot_finish, unnormal_lot_start,
                                unnormal_lot_finish, columns)

            st.title("로트별 데이터")
            st.pyplot(fig)

    '''
    st.code(code, language='python')