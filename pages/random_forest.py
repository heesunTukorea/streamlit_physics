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
st.set_page_config(page_title="ramdom_forest")

# 화면 분할
col1, col2 = st.columns([7, 2])

with col2:
    st.image("tuk_img.png")

st.title('랜덤 포레스틈 모델')

# 사이드바의 선택 박스
selecop = st.sidebar.selectbox('카테고리', ('비정상 데이터','random_forest'))

df_QC = add_QC(dedicated_data)
sensor_data_df = control_sensor_data(df_QC)
correlation_sen = correlation_def1(sensor_data_df)

if selecop == '비정상 데이터':
    st.title("비정상 데이터")
    st.write("임계치를 넘는 데이터")
    st.dataframe(sensor_data_df)
    st.write(sensor_data_df.shape)
    correlation_sen = correlation_def1(sensor_data_df)
    st.title("상관관계 분석")
    st.write(correlation_sen)
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_sen, annot=True, fmt=".2f", ax=ax1)
    st.pyplot(fig1)

    


new_df = calculator_data(df_list,df_list2)
correlation_sen = correlation_def1(sensor_data_df)

# 의사결정나무 분류 선택
if selecop == 'random_forest':
    st.title("랜덤포레스트 분류 모델")
    col8, col9, col10 = st.columns([2, 2, 2])
    with col8:
        class_weight = st.number_input("class_weight", value=0.02, step=0.01)
    with col9:
        max_depth_selected = st.checkbox("Select Max Depth")
        if max_depth_selected:
            max_depth = st.number_input("Max Depth", value=1, min_value=1, max_value=100)
        else:
            max_depth = None
    with col10:
        threshold = st.number_input("Threshold", value=0.5, step=0.1)
    
    split_size = st.slider('트레이닝셋 비율', 0.0, 1.0, value = 0.7, step =0.1)
    # Decision Tree 모델 학습
    if st.button('설정 완료'):
        class_ratio, accuracy,confusion_matrix_df, fm, fm_vi,rdf_fig = ramdom_forest_model(df_QC,sensor_data_df,threshold,split_size,class_weight,max_depth)
        st.write("추천 weight:", class_ratio)
        if fm is not None:
            st.success("랜덤포레스트 모델 학습이 완료되었습니다!")
            st.write("모델 정보:")
            st.write(fm)

            #st.pyplot(decision_tree_fig)
            st.image("Random_forest.png")            

        else:
            st.warning("랜덤 포레스트 모델 학습에 실패했습니다.")

        st.write("랜덤 포레스트 모델 평가\n")
        st.write("Accuracy:", accuracy)
        st.write("Confusion Matrix:")
        st.dataframe(confusion_matrix_df)
        st.pyplot(fm_vi)

        # st.subheader("오류값")
        # df_list=split_dataframe_by_lot(y_pred_adjusted,split_rate,sensor_data_df)
        # new_error_df = lot_data2(df_list)
        # st.dataframe(new_error_df)



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
  


    

