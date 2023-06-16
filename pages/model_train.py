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
selecop = st.sidebar.selectbox('카테고리', ('데이터 보기', 'decision_Tree','decision_Tree_Test'))

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



# 의사결정나무 선택
if selecop == 'decision_Tree':
    global_model = None
    st.title("의사결정 나무 모델 학습")
    col3, col4, col5, col6, col7 = st.columns([2, 2, 2, 2, 2])
    with col3:
        max_trials = st.number_input("최대 모델 탐색 횟수", value=5)
    with col4:
        epochs = st.number_input("훈련 에포크 수", value=7)
    with col5:
        verbose = st.number_input("훈련 과정의 상세도", value=2)
    with col6:
        is_training = st.checkbox("모델을 훈련할지 여부", value=True)
    with col7:
        vis = st.checkbox("시각화 여부", value=True)
    # Decision Tree 모델 학습
if st.button('설정 완료'):
    clf, model, train_data, test_data = DecisionTreeRegressor(new_df, 0.2, max_trials, epochs, verbose, is_training, vis)

    if clf is not None:
        st.success("의사결정 나무 모델 학습이 완료되었습니다!")
        st.write("모델 정보:")
        st.write(model)

        # Check if the model is trained
        if is_training:
            # Fit the model with training data
            clf.fit(train_data[["Min_pH", "Min_Temp", "Min_Voltage", "Min_Current",
                                "Max_pH", "Max_Temp", "Max_Voltage", "Max_Current", 
                                "Mean_pH", "Mean_Temp", "Mean_Voltage", "Mean_Current"]],
                    train_data["QC"])

        # Evaluate the model
        st.title("의사결정 나무 모델 평가")
        evaluate_regression_model(test_data,model)

        if vis:
           if vis:
                plt.figure(figsize=(10, 30))
                tree.plot_tree(clf, filled=True)
                plt.title('Decision Tree Visualization')
                plt.savefig("decision_tree.png")  # Save the decision tree visualization as an image
                plt.show()

                # Display the image in Streamlit
                st.image("decision_tree.png")
    else:
        st.warning("의사결정 나무 모델 학습에 실패했습니다.")



       
    st.title("의사결정 나무 모델 평가")
    y_test, y_pred, TP, FP, FN, TN, rmse, accuarcy_t_score, recall_t_score, precision_t_score, F1_t_score, tpr, fpr, tpr_val, fpr_val, auc_val = evaluate_regression_model(test_data, model)

    st.write("의사결정 나무 모델 평가\n")
    st.write("RMSE:", rmse)
    st.write("Accuracy:", accuarcy_t_score)
    st.write("Recall:", recall_t_score)
    st.write("Precision:", precision_t_score)
    st.write("F1 score:", F1_t_score)
    st.write("TP:", TP)
    st.write("FP:", FP)
    st.write("FN:", FN)
    st.write("TN:", TN)
    st.write("TPR (True Positive Rate):", tpr_val)
    st.write("FPR (False Positive Rate):", fpr_val)
    st.write("AUC Score:", auc_val)
    st.write("tpr:", tpr)
    st.write("fpr:", fpr)

    plt.figure()
    plt.plot(fpr, tpr, 'o-', label="Decision_Tree")
    plt.plot([0, 1], [0, 1], 'k--', label="random guess")
    plt.plot([fpr_val], [tpr_val], 'ro', ms=10)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.grid()
    plt.legend()


    # Display the plot in Streamlit
    st.pyplot(plt)

# # decision_Tree_Test 선택
# #if selecop == 'decision_Tree_Test':
#         #if st.button("테스트 결과 보기"):   
#         st.title("의사결정 나무 모델 평가")
#         evaluate_regression_model_streamlit(model.get_params(), test_data)

    

