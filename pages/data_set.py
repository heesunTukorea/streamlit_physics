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



# CSV 파일의 리스트만 추출합니다











# with st.sidebar:
#     selecop = st.selectbox('설정 메뉴',
#     ('job.csv', 'setup.csv', 'sim.csv'))

#     if selecop == 'job.csv':
#         st.subheader('job.csv파일 생성 설정')

#         number = st.number_input('job의 갯수')
#         number = int(number)
#         value1,value2 = st.slider('job타입의 난수범위',0,100,(1,10))
#         st.write('선택범위', value1,value2)

#         if st.button('job.csv생성'):
#             job(number,value1,value2)
#             st.write('생성완료')

#     elif selecop == 'setup.csv':
#         st.subheader('set.csv파일 생성 설정')

#         value3,value4 = st.slider(
#         'setup시간 난수의 범위',0,100,(1,10))
#         st.write('선택범위', value3,value4)

#         if st.button('setup.csv생성'):
#             setup(value3,value4)
#             st.write('생성완료')

#     elif selecop == 'sim.csv':
#         st.subheader('sim.csv파일 생성 설정')

#         number2 = st.number_input('기계의 갯수')
#         number2 = int(number2)
#         value5,value6 = st.slider('processtime의 난수범위',0,100,(1,10))
#         st.write('선택범위', value5,value6)

#         value7,value8 = st.slider(
#         'job_operation의 난수범위',0,100,(1,10))
#         st.write('선택범위', value7,value8)

#         if st.button('sim.csv생성'):
#             sim(number2,value5,value6,value7,value8)
#             st.write('생성완료')
        
    


# Define your functions for loading and processing data here...


# j_count=0
# s_count=0
# p_count=0

# if os.path.exists("FJSP_Job.csv"):
#     j_count +=1
# if os.path.exists("FJSP_Set.csv"):
#     s_count +=1
# if os.path.exists("FJSP_Sim.csv"):
#     p_count +=1 
# c_sum=j_count+s_count+p_count

# if c_sum==3:
#     tab1, tab2, tab3, tab4 = st.tabs(["job.csv", "setup.csv", "process_time.csv", "q_time.csv"])
#     with tab1:
#         st.header("job.csv")
#         st.write(job_df)
#     with tab2:
#         st.header("setup.csv")
#         st.write(setup_df)
#     with tab3:
#         st.header("process_time.csv")
#         st.write(sim_df)
#     with tab4:
#         st.header("q_time.csv")
# else:
#     st.write("데이터 정보를 입력해 주세요")
