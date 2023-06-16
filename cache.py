import datetime
from matplotlib import pyplot as plt
from sklearn import tree
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
# def evaluate_regression_model(model, test_data):
#     """
#     회귀 모델을 평가하고 결과를 출력합니다.

#     Parameters:
#         model: 평가할 회귀 모델
#         test_data (pandas.DataFrame): 테스트 데이터

#     Returns:
#         None
#     """
#     # AutoKeras 모델 예측
#     ak_model_predicted = model.predict(test_data[["Min_pH", "Min_Temp", "Min_Voltage", "Min_Current",
#                                                   "Max_pH", "Max_Temp", "Max_Voltage", "Max_Current",
#                                                   "Mean_pH", "Mean_Temp", "Mean_Voltage", "Mean_Current"]])
#     print("AutoKeras 모델 예측\n")
#     print('AutoKeras Model Predict:', ak_model_predicted)
#     print("\n --------------------------------------\n")
    
#     # RMSE 계산
#     rmse = sqrt(mean_squared_error(test_data['QC'], ak_model_predicted))
#     print("RMSE 계산\n")
#     print('AutoKeras Model RMSE:', rmse)

#     # 분류 모델 평가
#     y_test = test_data['QC']
#     y_pred = [round(y[0], 0) for y in ak_model_predicted]
#     print("분류 모델 평가\n")
#     print("Accuracy =", accuracy_score(y_test, y_pred))
#     print("Recall =", recall_score(y_test, y_pred))
#     print("Precision =", precision_score(y_test, y_pred))
#     print("F1 score =", f1_score(y_test, y_pred))
#     print("\n --------------------------------------\n")

#     # Confusion Matrix 계산
#     def get_confusion_matrix_values(y_true, y_pred):
#         cm = confusion_matrix(y_true, y_pred)
#         return cm[0][0], cm[0][1], cm[1][0], cm[1][1]

#     TP, FP, FN, TN = get_confusion_matrix_values(y_test, y_pred)

#     print("Confusion Matrix 계산\n")
#     print("TP:", TP)
#     print("FP:", FP)
#     print("FN:", FN)
#     print("TN:", TN)
#     print("\n --------------------------------------\n")

#     # TPR, FPR 계산
#     if (TP + FN) == 0:
#         tpr_val = 0
#     else:
#         tpr_val = TP / (TP + FN)

#     if (TN + FP) == 0:
#         fpr_val = 0
#     else:
#         fpr_val = TN / (TN + FP)
    
#     print("TPR, FPR 계산\n")
#     print("TPR:", tpr_val)
#     print("FPR:", fpr_val)
#     print("\n --------------------------------------\n")

#     # ROC Curve 계산 및 플롯
#     tpr, fpr, _ = roc_curve(y_test, y_pred)
#     tpr[1] = tpr_val
#     fpr[1] = fpr_val

#     if len(tpr) < 3:
#         tpr = np.append(tpr, 1)
#         fpr = np.append(fpr, 1)
        
#     print("ROC Curve 계산 및 플롯\n")
#     print("FPR:", fpr)
#     print("TPR:", tpr)
#     print("\n --------------------------------------\n")

#     plt.plot(tpr, fpr, 'o-', label="Logistic Regression")
#     plt.plot([0, 1], [0, 1], 'k--', label="random guess")
#     plt.plot([tpr_val], [fpr_val], 'ro', ms=10)
#     plt.xlabel('Fall-Out')
#     plt.ylabel('Recall')
#     plt.title('Receiver operating characteristic example')
#     plt.grid()
#     plt.legend()
#     plt.show()

#     # Classification Report 출력
#     print("\n --------------------------------------\n")
#     print(classification_report(y_test, y_pred, target_names=['class 0','class 1']))