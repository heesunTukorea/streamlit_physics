import streamlit as st
from PIL import Image

st.set_page_config(page_title="main_page")

st.markdown(
    """
    <style>
    body {
        background-color: #F5F5F5;
    }
    </style>
    """,
    unsafe_allow_html=True
)
col1, col2 = st.columns([7, 2])

with col1:
    st.title("도금욕 공정의 센서 데이터를 이용한 품질 예측")

    


with col2:
    image = Image.open("tuk_img.png")
    st.image(image)

with st.sidebar:
    st.sidebar.markdown("## 목록")
    st.write("총괄리더: 신형찬")
    st.write("분석팀장: 안이찬")
    st.write("개발팀장: 양희선")
    st.write("팀원: 정원석")


st.header("")
st.header("")
st.header("")



image1 = Image.open("dogum.png")
st.image(image1)