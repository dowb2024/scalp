import streamlit as st
from datetime import date

# 배경색 설정을 위한 CSS
st.markdown(
    """
    <style>
    .stApp {
        background-color: #ffffff;
    }

    /* 페이지 좌우 여백 줄이기 */
    .main > div {
        max-width: 80%; /* 기본값은 80%입니다. 필요한 만큼 넓힐 수 있습니다 */
        padding-left: 5%;
        padding-right: 5%;
    }
    .custom-text {
        padding-left: 20px; /* 좌측 여백 */
        padding-right: 20px; /* 우측 여백 */
        text-align: left;
    }

    .highlight-box {
        background-color: #d2b48c; /* 노란색 박스 */
        padding: 15px;
        border-radius: 10px;
        border: 2px solid #d2b48c;
        font-weight: bold;
        text-align: center;
        color: #333;
        font-size: 15px;
        margin-top: 0px;
        margin-bottom: 20px;
        width: 100%; /* 박스의 폭을 줄임 */
        margin-left: 0;
        margin-right: auto;
    }
    </style>
    """, unsafe_allow_html=True
)

# 오늘의 날짜 가져오기
today = date.today()

# 화면에 오늘 날짜 표시
st.image("./data/banner_1.jpg", use_column_width=False)
st.markdown(f"{today.strftime('%Y.%m.%d')}, made by DeepRoot(김성환, 김준호, 이혜진, 전민정)")


# 소개 섹션
st.markdown(
    """
    **소개 📝**
    \n두피 관리를 위해 병원을 방문하는 것은 대기 시간과 복잡한 절차로 번거로울 수 있어 많은 사람들이 소홀히 하기 쉽습니다.  
    그래서 저희는 EfficientNet으로 학습된 CNN 모델을 활용하여 두피 이미지를 분석하고 증상을 파악합니다.  
    덕분에 이미지 업로드만으로도 **빠르고 쉽게 두피 상태를 점검하고**, 개인의 두피 증상에 **최적화된 제품을 추천**받을 수 있습니다.  
    이를 통해 두피 문제를 사전에 감지하고 예방하며, 언제 어디서나 편리하게 두피 상태를 관리할 수 있습니다.
    """, unsafe_allow_html=True
)

st.markdown("""**이용 방법 🚀**""")

# 강조된 텍스트
st.markdown(
    """
    <div class="highlight-box">[두피 이미지 업로드] - [설문 조사] - [두피 진단] - [증상 원인 / 특징 / 관리방안 설명] - [제품 추천]</div>
    """, unsafe_allow_html=True
)

# 이용 방법 섹션
st.markdown(
    """
    **[두피 이미지 업로드] 📸**
    \n- 가이드라인에 맞춰 두피 사진을 찍은 후 **[Browse Files]** 버튼을 눌러 이미지를 업로드합니다.

**[설문 조사] 📝**
    \n - 자신의 두피와 가장 적합하다고 생각되는 타입을 정해주세요.
    \n - 가지고 계신 두피 고민이나 병원에서 진단받은 증상을 선택해주세요.
    \n - 저희에게 추천받고 싶은 제품의 카테고리를 정해주세요.

**[두피 진단] 🧠**
    \n- 업로드해주신 두피 이미지와 설문조사를 종합해 딥러닝으로 학습된 모델이 증상을 진단합니다.
    \n- 각 증상은 '비듬', '미세각질', '모낭사이홍반', '모낭홍반농포', '피지과다', '탈모' 중에서 진단하며, 각 진단 별로 심각도에 따라 **양호 / 경증 / 중증**으로 나누어 판단합니다.

**[증상 원인 / 특징 / 관리방안 설명] 💡**
    \n- 진단된 두피 증상의 원인 / 특징 / 관리 방안을 상세히 안내해드립니다.

**[제품 추천] 🎁**
    \n- 설문에서 선택한 제품 카테고리와 진단된 증상에 따라, 증상 개선에 도움이 되는 성분을 포함한 제품을 추천드립니다.
    \n- 많은 소비자들이 사용하고 높은 평가를 받은 제품 중 3가지를 소개해드립니다.
    """, unsafe_allow_html=True
)