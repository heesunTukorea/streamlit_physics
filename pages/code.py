import streamlit as st



col1, col2 = st.columns([7,2])
st.title("코드")
with col2:
    st.image("tuk_img.png")
with st.sidebar:
    selecop = st.selectbox('코드 선택', ('data_pre_processing.py', 'data_set.py','cache.py','logistic.py','dicision_tree.py'))
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

if selecop == 'data_pre_processing.py':

    code = '''
    import streamlit as st
    import pandas as pd
    import numpy as np
    from cache import load_csv, file_list
    import os
    import datetime
    from sklearn.model_selection import train_test_split
    import sklearn
    from matplotlib import pyplot as plt
    import seaborn as sns

    # Display editor's content as you type
    st.set_page_config(page_title="data_set")

    col1, col2 = st.columns([7, 2])

    with col2:
        st.image("tuk_img.png")

    st.title('데이터셋')

    st.title("Data Loading and Processing")



    with st.sidebar:
        selecop = st.selectbox('선택 csv', file_list)


    df = load_csv(selecop)
    st.dataframe(df)
    df_shape = df.shape
    st.write(" 데이터 형태:", df_shape)


    '''
st.code(code, language='python')


if selecop == 'cache.py':

    code = '''
        import datetime

    from matplotlib import pyplot as plt
    from sklearn import tree
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, auc


    from sklearn.metrics import classification_report, confusion_matrix, f1_score, mean_squared_error, precision_score, recall_score, roc_curve
    from sklearn.model_selection import train_test_split
    import streamlit as st
    import pandas as pd
    import numpy as np
    import os
    import seaborn as sns
    import tensorflow
    from sklearn import tree
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.decomposition import PCA
    from mpl_toolkits.mplot3d import Axes3D


    #csv파일 가져오기
    @st.cache_data
    def load_csv(data_csv_name):
        df = pd.read_csv(data_csv_name, index_col=0)
        
        return df


    #가져올 파일 리스트
    file_list = ['Error Lot list.csv', 'kemp-abh-sensor-2021.09.06.csv', 'kemp-abh-sensor-2021.09.07.csv', 
                'kemp-abh-sensor-2021.09.08.csv', 'kemp-abh-sensor-2021.09.09.csv', 'kemp-abh-sensor-2021.09.10.csv', 
                'kemp-abh-sensor-2021.09.13.csv', 'kemp-abh-sensor-2021.09.14.csv', 'kemp-abh-sensor-2021.09.15.csv', 'kemp-abh-sensor-2021.09.16.csv', 
                'kemp-abh-sensor-2021.09.17.csv', 'kemp-abh-sensor-2021.09.23.csv', 'kemp-abh-sensor-2021.09.24.csv', 'kemp-abh-sensor-2021.09.27.csv', 
                'kemp-abh-sensor-2021.09.28.csv', 'kemp-abh-sensor-2021.09.29.csv', 'kemp-abh-sensor-2021.09.30.csv', 'kemp-abh-sensor-2021.10.01.csv', 
                'kemp-abh-sensor-2021.10.05.csv', 'kemp-abh-sensor-2021.10.06.csv', 'kemp-abh-sensor-2021.10.07.csv', 'kemp-abh-sensor-2021.10.08.csv', 
                'kemp-abh-sensor-2021.10.12.csv', 'kemp-abh-sensor-2021.10.13.csv', 'kemp-abh-sensor-2021.10.14.csv', 'kemp-abh-sensor-2021.10.15.csv', 
                'kemp-abh-sensor-2021.10.18.csv', 'kemp-abh-sensor-2021.10.19.csv', 'kemp-abh-sensor-2021.10.20.csv', 'kemp-abh-sensor-2021.10.21.csv', 
                'kemp-abh-sensor-2021.10.22.csv', 'kemp-abh-sensor-2021.10.25.csv', 'kemp-abh-sensor-2021.10.26.csv','kemp-abh-sensor-2021.10.27.csv']

    # CSV 파일의 리스트만 추출합니다

    for i in file_list:
    load_csv(i)

    # 날짜,시간 전처리 코드
    @st.cache_data
    def load_and_merge_csvs(file_list):
        dfs = []
        for data_list in file_list[1:]:
            tmp = pd.read_csv(data_list, sep=',', encoding='utf-8')
            y, m, d = map(int, data_list.split('-')[-1].split('.')[:-1])

            time = tmp['Time']
            tmp['DTime'] = '-'.join(data_list.split('-')[-1].split('.')[:-1])
            ctime = time.apply(lambda _: _.replace(u'오후', 'PM').replace(u'오전', 'AM'))
            n_time = ctime.apply(lambda _: datetime.datetime.strptime(_, "%p %I:%M:%S.%f"))
            newtime = n_time.apply(lambda _: _.replace(year=y, month=m, day=d))
            tmp['Time'] = newtime

            dfs.append(tmp)

        merged_df = pd.concat(dfs, axis=0, ignore_index=True)
        merged_df = merged_df.drop('Index', axis=1)
        merged_df = merged_df.set_index('Time')

        return merged_df
    load_and_merge_csvs(file_list)

    merged_df = load_and_merge_csvs(file_list)


    dedicated_data = merged_df.copy()


    #QC : 정상 비정상 할당 코드
    @st.cache_data
    def add_QC(dedicated_data):
    error_list = "Error Lot list.csv"
    #Lot List추출
    #unique()는 칼럼에 있는 값의 유일 값 추출
    lot_lists = dedicated_data['Lot'].unique()

    #date list 추출
    d_lists = dedicated_data['DTime'].unique()

    #Error data read
    error = pd.read_csv(error_list, sep=',', encoding='utf-8')

    #결측치를 가진 행 제거
    error_drop = error.dropna()

    #Error data List추출
    lot_error_lists = error_drop['LoT'].unique()
    d_error_lists = error_drop['Date'].unique()

    #X_data = pd.DataFrame(columns=[pH','Temp','Current','Voltage','QC'])
    #두개의 데이터프레임에서 필요한 부분만 합치는 코드->날짜와 로트 뽑아놓은 리스트와 같은것
    X_data = pd.DataFrame()
    for d in d_lists:#날짜 리스트
        for lot in lot_lists:#로트 리승트
            #dd= 원본데이터, 원본데이터의 날짜와 로트가  뽑아놓은 리스트와 같은것을 필터링
            tmp = merged_df[(merged_df['DTime'] == d) & (merged_df['Lot'] == lot)]
            tmp = tmp[['pH', 'Temp', 'Current', 'Voltage','DTime','Lot']]
            
            #error_drop은 결측치를 제거한 에러리스트, 
            error_df = error_drop[(error_drop['Date'] == d) & (error_drop['LoT'] == lot)]
            len_error = len(error_df)
            #QC라는 en_error >0이면 0으로 채우고 아니라면 1로 채움
            if len_error > 0:
                trr = np.full((tmp['pH'].shape), 0)
            else:
                trr = np.full((tmp['pH'].shape), 1)
            tmp['QC'] = trr
            #X_data = X_data.append(tmp)
            X_data = pd.concat([X_data, tmp]) # 코드 수행이 안되서 바꿈

    #X_data.set_index(["Lot", "DTime"], inplace=True)
    return X_data

    X_data = add_QC(dedicated_data)


    #상관관계 분석 코드
    @st.cache_data
    def correlation_def(cor_df):
    if 'DTime' in cor_df.columns:
            cor_df1 = cor_df.drop('DTime', axis=1)
            correlation = cor_df1.corr()
            return correlation
    else:
        st.error("Column 'DTime' not found in the DataFrame.")
        return None
    correlation_def(dedicated_data)
    correlation_def(X_data)

    @st.cache_data
    def correlation_def1(cor_df):

        correlation = cor_df.corr()
        return correlation




    #로트 단위로 정상 비정상 데이터를 나누는 코드
    @st.cache_data
    def sensors_of_lots(data):
        dedicated_data1 = data

        data_unnormal = dedicated_data1[dedicated_data1["QC"]== 0] #이상 데이터
        data_normal = dedicated_data1[dedicated_data1["QC"]== 1] #정상 데이터
        data_normal_unique = data_normal.groupby(["DTime", "Lot"]).size().reset_index(name='Count') #키 값 생성
        data_unnormal_unique = data_unnormal["DTime"].unique() #키 값 생성
        #data_unnormal_unique = data_unnormal["DTime"].unique()


        df_list = []  # 정상 로트 저장
        df_list2 = []  # 비정상 로트 저장

        for i, row in data_normal_unique.iterrows():
            dt = row["DTime"]
            lot = row["Lot"]
            df_subset = data_normal[(data_normal["DTime"] == dt) & (data_normal["Lot"] == lot)]
            df_list.append(df_subset)

        for dt in data_unnormal_unique:
        df_subset = data_unnormal[data_unnormal["DTime"] == dt]
        df_list2.append(df_subset)

        columns = ["Temp", "Voltage", "pH", "Current"]
        num_columns = len(columns)
        return df_list, df_list2, columns
        # Dtime에 따라 데이터프레임 분할


    

        
    df_list, df_list2, columns = sensors_of_lots(X_data)

    #로트 단위로 정상 비정상 할당해 시각화 하는 코드
    @st.cache_data
    def plot_in_sensors(data, df_list, df_list2, normal_lot_start, normal_lot_finish, unnormal_lot_start, unnormal_lot_finish, columns):
        plt.figure(figsize=(15, 8))
        #쓰레쉬 홀드 계산
        threshold = []
        columns = ["Temp", "Voltage", "pH", "Current"]
        des = dedicated_data.describe()
        for column in columns:
            # 정상 데이터의 평균과 표준편차 계산
            normal_mean = des[column]["mean"]
            normal_std = des[column]['std']

            # 쓰레쉬홀드 설정 (평균 값의 표준편차를 +-로 설정)
            threshold.append((normal_mean - 3*normal_std, normal_mean + 3*normal_std))
    
        if unnormal_lot_finish != "없음":
            for i, column in enumerate(columns):
                plt.subplot(2, 2, i+1)  # 2x2 subplot 중 현재 subplot 설정

            # 각 데이터프레임 별로 plot 그리기
                for j, df_subset in enumerate(df_list2[unnormal_lot_start: unnormal_lot_finish]):
                    x = [k for k in range(len(df_subset))]
                    y = df_subset[column]

                    plt.plot(x, y, label=f"abnormal {j+1}")
                plt.xlabel("Time")
                plt.ylabel(column)

                plt.title(column)
                plt.legend()

        if normal_lot_finish != "없음":
            for i, column in enumerate(columns):
                plt.subplot(2, 2, i+1)  # 2x2 subplot 중 현재 subplot 설정

                # 각 데이터프레임 별로 plot 그리기
                for j, df_subset in enumerate(df_list[normal_lot_start: normal_lot_finish]):
                    x = [k for k in range(len(df_subset))]
                    y = df_subset[column]

                # 쓰레쉬홀드 그리기
                    plt.plot(x, [threshold[i][0]] * len(df_subset), "--", color="red", alpha=0.7)
                    plt.plot(x, [threshold[i][1]] * len(df_subset), "--", color="red", alpha=0.7)

                    plt.plot(x, y, label=f"normal {j+1}")

                plt.xlabel("Time")
                plt.ylabel(column)

                plt.title(column)
                plt.legend()

        plt.tight_layout()  # subplot 간격 조절

        return plt.gcf() # 수정된 부분: plot 객체를 반환합니다.




    #계산된 데이터 프레임 만드는 코드
    @st.cache_data
    def calculator_data(df_list,df_list2):
        # 새로운 데이터프레임을 저장할 리스트
        new_data = []
        df_list3 = df_list + df_list2
        # 각 데이터프레임 별로 최대, 최소, 평균 값을 구하고 새로운 데이터프레임에 추가
        for df in df_list3:
            max_pH = df["pH"].max()
            max_Temp = df["Temp"].max()
            max_Voltage = df["Voltage"].max()
            max_Current = df["Current"].max()

            min_pH = df["pH"].min()
            min_Temp = df["Temp"].min()
            min_Voltage = df["Voltage"].min()
            min_Current = df["Current"].min()

            mean_pH = df["pH"].mean()
            mean_Temp = df["Temp"].mean()
            mean_Voltage = df["Voltage"].mean()
            mean_Current = df["Current"].mean()

            Lot = df["Lot"].iloc[0]
            DTime = df["DTime"].iloc[0]
            QC = df["QC"].max()

            new_data.append([Lot, DTime, max_pH, min_Temp, min_Voltage, min_Current, mean_pH, max_Temp, max_Voltage,
                            max_Current, mean_Temp, mean_Voltage, mean_Current,min_pH, QC])

        # 새로운 데이터프레임 생성
            new_df = pd.DataFrame(new_data, columns=["Lot", "DTime","Min_pH", "Min_Temp", "Min_Voltage", "Min_Current",
                                                "Max_pH", "Max_Temp", "Max_Voltage", "Max_Current", 
                                                "Mean_pH", "Mean_Temp", "Mean_Voltage", "Mean_Current", "QC"])
            new_df.set_index(["Lot", "DTime"], inplace=True)
        return new_df
    new_df = calculator_data(df_list,df_list2)

    #unnomal_df생성 코드-> 임계치를 넘는 값들의 것
    @st.cache_data
    def control_sensor_data(data):
        columns = ["Temp", "Voltage", "pH", "Current"]
        des = data.describe()
        # 쓰레쉬홀드 계산
        threshold = []
        for column in columns:
            # 정상 데이터의 평균과 표준편차 계산
            normal_mean = des[column]["mean"]
            normal_std = des[column]['std']

            # 쓰레쉬홀드 설정 (평균 값의 표준편차를 +-로 설정)
            threshold.append((normal_mean - 3*normal_std, normal_mean + 3*normal_std))
        print(threshold)

        sensor_data = []

        for i, _ in data.iterrows():
            for j, column in enumerate(columns):
                sensor = data.loc[i]  # i번째 행, column 컬럼 데이터 가져오기
                sensor_col = sensor[column]
                if sensor_col <= threshold[j][0] or sensor_col >= threshold[j][1]:
                    #print(i,column,sensor_col,sensor["error"])
                    sensor_data.append(sensor)
                    break
        sensor_data_df = pd.DataFrame(sensor_data)            
        return sensor_data_df


    #정상로트를 저장하는 함수, df = df_QC
    def split_dataframe_by_lot(y_pred_adjusted, split_size, df_test):
        ratio = split_size
        df = df_test.sample(frac=ratio)  # Randomly sample a fraction of the DataFrame
        df["pred"] = y_pred_adjusted
        df_list = []
        lot_unique = df.groupby(["DTime", "Lot"]).size().reset_index(name='Count')

        for _, row in lot_unique.iterrows():
            dt = row["DTime"]
            lot = row["Lot"]
            df_subset = df[(df["DTime"] == dt) & (df["Lot"] == lot)]
            df_list.append(df_subset)

        return df_list

    # 새로운 에러 데이터프레임 생성
    def lot_data2(df_list):
        # 새로운 데이터프레임을 저장할 리스트
        new_error_data = []
        
        # 각 데이터프레임별로 최대, 최소, 평균 값을 구하고 새로운 데이터프레임에 추가
        for df in df_list:
            error = df["QC"].max()
            number_of_error = len(df)
            
            number_of_true = 0
            number_of_false = 0
            
            for index, row in df.iterrows():
                if row["QC"] == row["pred"]: 
                    number_of_true += 1
                else:
                    number_of_false += 1
            
            Lot = df["Lot"].iloc[0]
            DTime = df["DTime"].iloc[0]
            new_error_data.append([Lot, DTime, error, number_of_error, number_of_true, number_of_false])

        # 새로운 데이터프레임 생성
        new_error_df = pd.DataFrame(new_error_data, columns=["Lot", "DTime", "error_type", "number_of_error", "number_of_true", "number_of_false"])
        new_error_df.set_index(["Lot", "DTime"], inplace=True)

        # 새로운 데이터프레임 반환
        return new_error_df

    #센서데이터 시각화를 위한 함수 : unnormal_data
    @st.cache_data
    def process_data(data):
        data = data.copy()  # Make a copy of the data to avoid modifying the original DataFrame

        # Processing columns
        columns = ["Temp", "Voltage", "pH", "Current"]

    # 열에 대한 통계 계산
        des = data.describe()

        # 각 열의 평균을 사용하여 임계치 계산
        thresholds = []
        for column in columns:
            # 정상 데이터의 평균과 표준편차 계산
            normal_mean = des[column]["mean"]
            normal_std = des[column]['std']

            # 임계치 설정 (평균 값의 표준편차를 +-로 설정)
            thresholds.append((normal_mean - 3*normal_std, normal_mean + 3*normal_std))

        # 이상치가 있는 센서 데이터를 저장할 리스트
        sensor_data = []

        # 이상치가 있는 열과 해당 값을 저장할 리스트
        col = []

        # 각 행의 데이터를 반복하면서 이상치 처리
        for i, row in data.iterrows():
            k = -1  # 인덱스가 존재하지 않는 값으로 k를 초기화합니다.
            for j, column in enumerate(columns):
                sensor_col = row[column]
                if sensor_col <= thresholds[j][0] or sensor_col >= thresholds[j][1]:
                    #print(i, column, sensor_col, row["QC"], i % 68 + 1)
                    col.append([i, column, sensor_col, row["QC"], (i % 68) + 1])
                    if k != j:
                        sensor_data.append(row)
                        k = j

        return sensor_data, col

    #히스토그램
    @st.cache_data
    def process_error_data_hist(col):
        # 변수별 전체 카운트를 저장할 딕셔너리
        variable_all_counts = {}

        # 변수별 에러 카운트를 저장할 딕셔너리
        variable_error_counts = {'Temp': 0, "Current": 0, "pH": 0, "Voltage": 0}

        # 변수별 정상 카운트를 저장할 딕셔너리
        variable_counts = {'Temp': 0, "Current": 0, "pH": 0, "Voltage": 0}

        # 에러가 있는 경우의 시간을 저장할 리스트
        variable_error_time = []

        # 정상적인 경우의 시간을 저장할 리스트
        variable_time = []

        # 주어진 리스트(col)를 반복하면서 변수별로 카운트 및 시간을 계산
        for _, variable, _, error, time in col:
            if variable in variable_all_counts:
                variable_all_counts[variable] += 1
                if error == 0:
                    variable_counts[variable] += 1
                    variable_time.append(time)
                else:
                    variable_error_counts[variable] += 1
                    variable_error_time.append(time)
            else:
                variable_all_counts[variable] = 1

        # 결과 출력
        print(variable_all_counts)
        print(variable_counts)
        print(variable_error_counts)

        # 히스토그램 그리기
        x1 = [i for i in range(len(variable_error_time))]
        x2 = [i for i in range(len(variable_time))]

        plt.hist(variable_error_time, bins=20, edgecolor='black')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title('Histogram')
        plt.show()

        # 결과 반환
        return plt.gcf(),variable_all_counts, variable_counts, variable_error_counts, variable_error_time, variable_time

    #파이차트 함수
    @st.cache_data
    def plot_pie_chart(variable_counts):
        # 변수별 카운트를 기반으로 데이터 준비
        labels = variable_counts.keys()
        values = variable_counts.values()

        # 파이차트 그리기
        plt.pie(values, labels=labels, autopct='%1.1f%%')
        plt.title('Variable Error Counts')

        # 보기 좋게 조정
        plt.axis('equal')
        plt.show()

        return plt.gcf()



    #sensor_data = control_sensor_data(X_data)

    @st.cache_data
    def train_split(new_df,test_size):
        train_data, test_data = train_test_split(new_df, test_size)
        return train_data, test_data, test_size

    global_model = None

    # 의사결정나무 테스트 코드
    @st.cache_resource
    def DecisionTreeRegressor(data, test_size, max_trials, epochs, verbose, is_training, vis):
        """
        의사결정나무 회귀 모델을 구축하고 시각화합니다.

        Parameters:
            data (pandas.DataFrame): 모델링에 사용할 데이터
            test_size (float): 테스트 데이터의 비율 (기본값: 0.2)
            max_trials (int): AutoKeras의 최대 모델 탐색 횟수 (기본값: 5)
            epochs (int): 훈련 에포크 수 (기본값: 7)
            verbose (int): 훈련 과정의 출력 상세도 (기본값: 2)
            is_training (bool): AutoKeras 모델을 훈련할지 여부 (기본값: True)
            vis (bool): 의사결정나무 시각화 여부 (기본값: True)

        Returns:
            clf (DecisionTreeRegressor): 훈련된 의사결정나무 회귀 모델
            model (AutoKeras Model): 훈련된 AutoKeras 구조
            train_data (pandas.DataFrame): 훈련 데이터
            test_data (pandas.DataFrame): 테스트 데이터
        """
        
        # 트레이닝 테스트 데이터 나누기
        train_data, test_data = train_test_split(data, test_size=test_size)
        
        
        # 의사결정나무 모델 구축
        clf = tree.DecisionTreeRegressor()
            
        # 의사결정나무 모델 학습
        model = clf.fit(train_data[["Min_pH", "Min_Temp", "Min_Voltage", "Min_Current",
                            "Max_pH", "Max_Temp", "Max_Voltage", "Max_Current", 
                            "Mean_pH", "Mean_Temp", "Mean_Voltage", "Mean_Current"]],
                train_data[['QC']])
        
        # 의사결정나무 모델 시각화 
        if vis:
            plt.figure(figsize=(10, 30)) 
            tree.plot_tree(clf)
            plt.savefig("decision_tree.png")  # 시각화한 의사결정나무를 파일로 저장
            plt.show()
        global global_model
        global_model = model
        return model, clf, train_data, test_data



    # 의사결정 나무 테스트 코드
    @st.cache_resource
    def evaluate_regression_model(test_data, _model):
        """
        회귀 모델을 평가하고 결과를 출력합니다.

        Parameters:
            test_data (pandas.DataFrame): 테스트 데이터
            model (sklearn.tree.DecisionTreeRegressor): 평가할 의사결정 나무 모델

        Returns:
            Tuple or None: 평가 결과를 담은 튜플 또는 None
        """
        if _model is None:
            print("모델이 초기화되지 않았습니다. 모델을 훈련시키고 다시 시도해주세요.")
            return None

        # 의사결정 나무 모델 예측
        ak_model_predicted = _model.predict(test_data[["Min_pH", "Min_Temp", "Min_Voltage", "Min_Current",
                                        "Max_pH", "Max_Temp", "Max_Voltage", "Max_Current",
                                        "Mean_pH", "Mean_Temp", "Mean_Voltage", "Mean_Current"]])

        y_pred = [round(y, 0) for y in ak_model_predicted]
        y_test = test_data['QC']
        # RMSE 계산
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        # 분류 모델 평가
        
        accuarcy_t_score = accuracy_score(y_test, y_pred)
        recall_t_score = recall_score(y_test, y_pred)
        precision_t_score = precision_score(y_test, y_pred)
        F1_t_score = f1_score(y_test, y_pred)

        # Confusion Matrix 계산
        TP, FP, FN, TN = confusion_matrix(y_test, y_pred).ravel()

        # TPR, FPR 계산
        if (TP + FN) == 0:
            tpr_val = 0
        else:
            tpr_val = TP / (TP + FN)

        if (TN + FP) == 0:
            fpr_val = 0
        else:
            fpr_val = FP / (TN + FP)

        # ROC Curve 계산
        fpr, tpr, _ = roc_curve(y_test, y_pred)
        tpr[1] = tpr_val
        fpr[1] = fpr_val
        # AUC 계산
        auc_val = auc(fpr, tpr)

        # 결과 출력
        print("의사결정 나무 모델 평가\n")
        print("RMSE:", rmse)
        print("Accuracy:", accuarcy_t_score)
        print("Recall:", recall_t_score)
        print("Precision:", precision_t_score)
        print("F1 score:", F1_t_score)
        print("TP:", TP)
        print("FP:", FP)
        print("FN:", FN)
        print("TN:", TN)
        print("TPR (True Positive Rate):", tpr_val)
        print("FPR (False Positive Rate):", fpr_val)
        print("AUC Score:", auc_val)

        return y_test, y_pred, TP, FP, FN, TN, rmse, accuarcy_t_score, recall_t_score, precision_t_score, F1_t_score, tpr, fpr, tpr_val, fpr_val,auc_val

    @st.cache_resource
    def pca_sensor_data(main_df, df):
        df2 = df.copy()
        
        # 데이터 정규화
        scaler = MinMaxScaler()
        df_scaled = scaler.fit_transform(df2)

        # PCA 모델 생성 및 적합
        pca = PCA(n_components=2)
        pca.fit(df_scaled)

        # 주성분 변환
        X_pca = pca.transform(df_scaled)

        # 주성분 변환된 데이터프레임 생성
        df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])

        # 변수의 영향력 계산
        var_exp = pca.explained_variance_ratio_
        components = pca.components_
        k = main_df["QC"].tolist()
        df_pca["QC"] = k

        return df_pca, components

    @st.cache_resource
    def plot_pca_3D(df_pca):
        error_types = df_pca['QC'].unique()
        colors = sns.color_palette('Set1', len(error_types))

        fig = plt.figure(figsize=(30, 10))
        ax = fig.add_subplot(111, projection='3d')

        # 에러 유형별 데이터 시각화
        for error_type, color in zip(error_types, colors):
            error_data = df_pca[df_pca['QC'] == error_type]
            ax.scatter(error_data['PC1'], error_data['PC2'], c=[color], alpha=0.5, label=error_type)

        ax.view_init(elev=30, azim=30)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_title('Clustering Results by Error Type')
        ax.legend()

        plt.show()
        return plt.gcf()


    #의사결정 나무 분류 코드
    #입력변수 =  가중치,깊이, 임계치


    @st.cache_resource
    def DecisionTreeClassifier_stream(data, data2, sensor_data_df, class_weight, max_depth, threshold, split_rate):
        # 데이터 준비
        df2 = data
        df2 = df2.reset_index()
        df = df2[["Max_pH", "Min_Temp", "Min_Voltage", "Min_Current", "Mean_pH", "Max_Temp", "Max_Voltage", "Max_Current", "Mean_Temp", "Mean_Voltage", "Mean_Current", "Min_pH"]]

        model = tree.DecisionTreeClassifier()

        test_df2 = sensor_data_df
        test_df = test_df2[['pH', 'Current', 'Temp', 'Voltage']]

        df3 = data2
        df4 = df3[["pH", "Current", "Temp", "Voltage"]]

        # 데이터 분할
        X = df4
        y = df3["QC"]

        X_t = test_df
        y_t = test_df2["QC"]
        split_rate_un = 1 - split_rate 
        X_train, X_test_dumi, y_train, y_test_dumi = train_test_split(X, y, train_size=split_rate, random_state= 12)
        X_train_dumi, X_test, y_train_dumi, y_test = train_test_split(X_t, y_t, test_size=split_rate_un, random_state= 12)
        
        # 클래스 가중치 설정
        class_counts = np.bincount(y)
        class_ratio = class_counts[0] / class_counts[1]
        class_weights = {0: 1.0, 1: class_weight}

        # 의사결정나무 모델 생성 및 학습
        model = tree.DecisionTreeClassifier(class_weight=class_weights, max_depth=max_depth)
        dts = model.fit(X_train, y_train)

        # Decision Tree 시각화
        fig, ax = plt.subplots(figsize=(10, 30))
        tree.plot_tree(dts, feature_names=X.columns, class_names=["0", "1"], filled=True, rounded=True)
        ax.set_title('Decision_Tree')
        decision_tree_fig = fig

        # 예측
        y_pred = model.predict(X_test)
        y_test = y_test.values

        fpr, tpr, _ = roc_curve(y_test, y_pred)

        # 양성 클래스(1)로 분류되는 경우의 확률만 가져옴
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # 임계값 기준으로 예측 결과 조정
        y_pred_adjusted = np.where(y_pred_proba >= threshold, 1, 0)

        # 정확도 평가
        accuracy = model.score(X_test, y_test)

        # 혼동 행렬
        cm = confusion_matrix(y_test, y_pred_adjusted)
        confusion_matrix_df = pd.DataFrame(cm, index=['실제 0', '실제 1'], columns=['예측 0', '예측 1'])

        # ROC 곡선
        fig, ax = plt.subplots(figsize=(16, 8))
        x = [i for i in range(len(y_pred_proba))]
        plt.plot(x, y_pred_proba)
        ax.set_title('ROC Curve')
        roc_curve_fig = fig

        

        decision_tree_fig_path = "Decision_Tree_Classifier.png"
        roc_curve_fig_path = "ROC_Curve.png"
        decision_tree_fig.savefig(decision_tree_fig_path)
        
        return y_test,y_pred_adjusted,decision_tree_fig, roc_curve_fig, X, dts, y_test, accuracy, y_pred, y_test, fpr, tpr, y_pred_proba, y_pred_adjusted, confusion_matrix_df, cm



    #로지스틱-> data= new_df
    @st.cache_resource
    def rogistic_model(data_new,main_data, class_weights, split_size,threshold):
        # 로지스틱 회귀 모델 학습 및 평가를 위한 함수

        # 테스트 데이터 준비
        data = main_data
        
        df2 = data_new
        df = df2[["Max_pH", "Min_Temp", "Min_Voltage", "Min_Current",
                                                "Mean_pH", "Max_Temp", "Max_Voltage", "Max_Current", "Mean_Temp",
                                                "Mean_Voltage", "Mean_Current","Min_pH"]]
        # 학습 데이터 준비
        
        X = df
        y = df2['QC']

        
        #최적 가중치 설정
        class_counts = np.bincount(y)
        class_ratio = class_counts[0] / class_counts[1]  # 비정상(1) 클래스에 대한 비율
        print(class_ratio)

        # 클래스 가중치 설정
        class_weights = {0: 1.0, 1: class_weights}

        # 데이터 분할 (학습 데이터와 테스트 데이터)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_size, random_state= 12)
        
        # 로지스틱 회귀 모델 생성 및 학습
        model = LogisticRegression(class_weight=class_weights)
        lo_model = model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_test2 = y_test.values

        fpr, tpr, _ = roc_curve(y_test, y_pred)
        
        # 테스트 데이터에 대한 예측 수행
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred_adjusted = np.where(y_pred_proba >= threshold, 1, 0)

    # 정확도 계산
        accuracy = model.score(X_test, y_test2)
        print("정확도:", accuracy)

        # 혼동 행렬 계산 및 출력
        cm = confusion_matrix(y_test2, y_pred_adjusted)
        confusion_matrix_df = pd.DataFrame(cm, index=['실제 0', '실제 1'], columns=['예측 0', '예측 1'])

        # 예측 확률에 대한 그래프 출력
        x = [i for i in range(len(y_pred_proba))]
        fig_lo = plt.figure(figsize=(16, 8))
        plt.plot(x, y_pred_proba, label="Pred")
        plt.plot(np.where(y_test2 == 0)[0], y_pred_proba[y_test2 == 0], 'bo', label="Test 0")
        plt.plot(x, [threshold] * len(y_pred_proba), "--", color="red", alpha=0.4)
        plt.plot(x, [0.5] * len(y_pred_proba), "--", color="red", alpha=0.4)
        plt.legend()

        return y_test2,y_pred_adjusted,class_ratio, accuracy, confusion_matrix_df, fig_lo, lo_model

    '''
st.code(code, language='python')

if selecop == 'logistic.py':

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
        st.title("데이터 보기")
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
            class_weight1 = st.number_input("class_weight", value=0.018, step=0.1)
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
        '''
st.code(code, language='python')

if selecop == 'dicision_tree.py':

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
        from sklearn import tree
        from sklearn.metrics import mean_squared_error, accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, roc_curve, classification_report


        # 페이지 할당
        st.set_page_config(page_title="model_train")

        # 화면 분할
        col1, col2 = st.columns([7, 2])

        with col2:
            st.image("tuk_img.png")

        st.title('모델학습과 성능 측정')

        # 사이드바의 선택 박스
        selecop = st.sidebar.selectbox('카테고리', ('비정상 데이터','decision_Tree'))

        df_QC = add_QC(dedicated_data)
        sensor_data_df = control_sensor_data(df_QC)
        correlation_sen = correlation_def1(sensor_data_df)

        if selecop == '비정상 데이터':
            st.title("비정상 데이터")
            st.dataframe(sensor_data_df)
            st.write(sensor_data_df.shape)
            correlation_sen = correlation_def1(sensor_data_df)
            st.title("상관관계 분석")
            st.write(correlation_sen)
            fig1, ax1 = plt.subplots(figsize=(10, 8))
            sns.heatmap(correlation_sen, annot=True, fmt=".2f", ax=ax1)
            st.pyplot(fig1)

            if st.button("더보기"):
                sensor_data, col = process_data(df_QC)
                fig,variable_all_counts, variable_counts, variable_error_counts, variable_error_time, variable_time = process_error_data_hist(col)
                st.title("히스토그램")
                st.pyplot(fig)

                labels = list(variable_counts.keys())
                values = list(variable_counts.values())

                st.title("파이차트")
                fig = plot_pie_chart(labels, values)
                st.pyplot(fig)


        new_df = calculator_data(df_list,df_list2)
        correlation_sen = correlation_def1(sensor_data_df)

        # 의사결정나무 분류 선택
        if selecop == 'decision_Tree':
            st.title("의사결정 나무 분류 모델")
            col8, col9, col10 = st.columns([2, 2, 2])
            with col8:
                class_weight = st.number_input("class_weight", value=20.0, step=0.1)
            with col9:
                max_depth_selected = st.checkbox("Select Max Depth")
                if max_depth_selected:
                    max_depth = st.number_input("Max Depth", value=1, min_value=1, max_value=100)
                else:
                    max_depth = None
            with col10:
                threshold = st.number_input("Threshold", value=0.5, step=0.1)
            
            split_rate = st.slider('트레이닝셋 비율', 0.0, 1.0, value = 0.7, step =0.1)
            # Decision Tree 모델 학습
            if st.button('설정 완료'):
                y_test,y_pred_adjusted,decision_tree_fig, roc_curve_fig, X, dts, y_test, accuracy, y_pred, y_test, fpr, tpr, y_pred_proba, y_pred_adjusted, confusion_matrix_df, cm = DecisionTreeClassifier_stream(new_df, X_data, sensor_data_df, class_weight, max_depth, threshold, split_rate)

                if dts is not None:
                    st.success("의사결정 나무 모델 학습이 완료되었습니다!")
                    st.write("모델 정보:")
                    st.write(dts)

                    #st.pyplot(decision_tree_fig)
                    st.image("Decision_Tree_Classifier.png")            

                else:
                    st.warning("의사결정 나무 모델 학습에 실패했습니다.")

                st.write("의사결정 나무 모델 평가\n")
                st.write("Accuracy:", accuracy)
                st.write("Confusion Matrix:")
                st.dataframe(confusion_matrix_df)
                st.pyplot(roc_curve_fig)

                st.subheader("오류값")
                df_list=split_dataframe_by_lot(y_pred_adjusted,split_rate,sensor_data_df)
                new_error_df = lot_data2(df_list)
                st.dataframe(new_error_df)



                # Display the plot in Streamlit











        # # 의사결정나무 선택
        # if selecop == 'decision_Tree':
        #     global_model = None
        #     st.title("의사결정 나무 모델 학습")
        #     col3, col4, col5, col6, col7 = st.columns([2, 2, 2, 2, 2])
        #     with col3:
        #         max_trials = st.number_input("최대 모델 탐색 횟수", value=5)
        #     with col4:
        #         epochs = st.number_input("훈련 에포크 수", value=7)
        #     with col5:
        #         verbose = st.number_input("훈련 과정의 상세도", value=2)
        #     with col6:
        #         is_training = st.checkbox("모델을 훈련할지 여부", value=True)
        #     with col7:
        #         vis = st.checkbox("시각화 여부", value=True)
        #     # Decision Tree 모델 학습
        #     if st.button('설정 완료'):
        #         clf, model, train_data, test_data = DecisionTreeRegressor(new_df, 0.2, max_trials, epochs, verbose, is_training, vis)

        #         if clf is not None:
        #             st.success("의사결정 나무 모델 학습이 완료되었습니다!")
        #             st.write("모델 정보:")
        #             st.write(model)

        #             # Check if the model is trained
        #             if is_training:
        #                 # Fit the model with training data
        #                 clf.fit(train_data[["Min_pH", "Min_Temp", "Min_Voltage", "Min_Current",
        #                                     "Max_pH", "Max_Temp", "Max_Voltage", "Max_Current", 
        #                                     "Mean_pH", "Mean_Temp", "Mean_Voltage", "Mean_Current"]],
        #                         train_data["QC"])

        #             # Evaluate the model
        #             st.title("의사결정 나무 모델 평가")
        #             evaluate_regression_model(test_data,model)

        #             if vis:
        #                 if vis:
        #                     plt.figure(figsize=(10, 30))
        #                     tree.plot_tree(clf, filled=True)
        #                     plt.title('Decision Tree Visualization')
        #                     plt.savefig("decision_tree.png")  # Save the decision tree visualization as an image
        #                     plt.show()

        #                     # Display the image in Streamlit
        #                     st.image("decision_tree.png")
        #         else:
        #             st.warning("의사결정 나무 모델 학습에 실패했습니다.")



                
        #         st.title("의사결정 나무 모델 평가")
        #         y_test, y_pred, TP, FP, FN, TN, rmse, accuarcy_t_score, recall_t_score, precision_t_score, F1_t_score, tpr, fpr, tpr_val, fpr_val, auc_val = evaluate_regression_model(test_data, model)

        #         st.write("의사결정 나무 모델 평가\n")
        #         st.write("RMSE:", rmse)
        #         st.write("Accuracy:", accuarcy_t_score)
        #         st.write("Recall:", recall_t_score)
        #         st.write("Precision:", precision_t_score)
        #         st.write("F1 score:", F1_t_score)
        #         st.write("TP:", TP, "FP:", FP, "FN:", FN, "TN:", TN)
            
        #         st.write("TPR (True Positive Rate):", tpr_val)
        #         st.write("FPR (False Positive Rate):", fpr_val)
        #         st.write("AUC Score:", auc_val)
        #         st.write("tpr:", tpr)
        #         st.write("fpr:", fpr)

        #         plt.figure()
        #         plt.plot(fpr, tpr, 'o-', label="Decision_Tree")
        #         plt.plot([0, 1], [0, 1], 'k--', label="random guess")
        #         plt.plot([fpr_val], [tpr_val], 'ro', ms=10)
        #         plt.xlabel('False Positive Rate')
        #         plt.ylabel('True Positive Rate')
        #         plt.title('Receiver operating characteristic example')
        #         plt.grid()
        #         plt.legend()


        #         # Display the plot in Streamlit
        #         st.pyplot(plt)

        with st.sidebar:
            st.sidebar.markdown("## 목록")
            st.write("총괄리더: 신형찬")
            st.write("분석팀장: 안이찬")
            st.write("개발팀장: 양희선")
            st.write("팀원: 정원석")
    


        


        '''
st.code(code, language='python')
with st.sidebar:
    st.sidebar.markdown("## 목록")
    st.write("총괄리더: 신형찬")
    st.write("분석팀장: 안이찬")
    st.write("개발팀장: 양희선")
    st.write("팀원: 정원석")