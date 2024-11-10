import streamlit as st
import torch
from PIL import Image
import os
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from torchvision import models, transforms
import pandas as pd
from openai import OpenAI
from matplotlib import pyplot as plt
import shutil
import json
from datetime import date
import httpagentparser
from streamlit_javascript import st_javascript
import streamlit.components.v1 as components

st.set_page_config(
    # layout="wide",
    page_title="두피케어 제품 추천 서비스",
    page_icon=".data/images/monsterball.png"
)

st.markdown(
    """
    <style>    
    .main > div {
        max-width: 80%; /* 기본값은 80%입니다. 필요한 만큼 넓힐 수 있습니다 */
        padding-left: 5%;
        padding-right: 5%;
    }   

    </style>
    """, unsafe_allow_html=True)
# /*
#     img {
#         max-height: 500px;
#     }
#     .streamlit-expanderContent div {
#         display: flex;
#         justify-content: center;
#         font-size: 20px;
#     }
#     [data-testid="stExpanderToggleIcon"] {
#         visibility: hidden;
#     }
#     .streamlit-expanderHeader {
#         pointer-events: none;
#     }
#     [data-testid="StyledFullScreenButton"] {
#         visibility: hidden;
#     }
#     */

# 오늘의 날짜 가져오기
today = date.today()

# 화면에 오늘 날짜 표시
st.image("./data/banner_1.jpg", use_column_width=True)
st.markdown(f"{today.strftime('%Y.%m.%d')}, made by DeepRoot(김성환, 김준호, 이혜진, 전민정)")


type_emoji_dict = {
    "(건성)두피에 건조함이나 당김을 느낍니다.": "🐲",
    "(지성)머리를 감은지 하루 이내에 두피가 기름집니다.": "🤖"
}

symptom_emoji_dict = {
    "비듬": "🐲",
    "미세각질": "🤖",
    "모낭사이홍반": "🧚",
    "모낭홍반농포": "🍃",
    "피지과다": "🔮",
    "탈모": "❄️"
}

variety_emoji_dict = {
    "샴푸": "🐲",
    "린스/컨디셔너": "🤖",
    "샴푸바/드라이샴푸": "🧚",
    "헤어오일/헤어세럼": "👨‍🚒",
    "헤어워터": "🦹",
    "두피팩/스케일러": "🦔",
    "헤어토닉/두피토닉": "🐯"
}

initial_scalp = [
    {
        "type": [""],
        "symptom": [""],
        "variety": [""],
        "bidum_state": "",
        "gakzil_state": "",
        "hongban_state": "",
        "nongpo_state": "",
        "pizy_state": "",
        "talmo_state": "",
        "result_type": []
    }
]

initial_upload = {
    "session": 0,
    "filepath": "",
    "filename": ""
}

example_scalps_img = [
    {
        "name": "정상",
        "url": "./data/images/nomal.jpg"
    },
    {
        "name": "비듬",
        "url": "./data/images/bidum.jpg"
    },
    {
        "name": "각질",
        "url": "./data/images/gakzil.jpg"
    },
    {
        "name": "홍반",
        "url": "./data/images/hongban.jpg"
    },
    {
        "name": "농포",
        "url": "./data/images/nongpo.jpg"
    },
    {
        "name": "피지",
        "url": "./data/images/pizy.jpg"
    },
    {
        "name": "탈모",
        "url": "./data/images/talmo.jpg"
    },
]


client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# 이미지 크기 조정 함수 추가
def resize_with_aspect_ratio(image, target_size):
    w, h = image.size
    aspect_ratio = w / h
    if aspect_ratio > 1:
        new_w = target_size
        new_h = int(target_size / aspect_ratio)
    else:
        new_h = target_size
        new_w = int(target_size * aspect_ratio)
    return image.resize((new_w, new_h), Image.BICUBIC)

@st.cache_data
def load_models():
    #사전 학습된 모델 불러오기 (24.10.30)
    model1 = torch.load('./data/models/bidum_model_label3_92.7.pt', map_location=torch.device('cpu'))
    model1.eval()  # 평가 모드로 전환 (드롭아웃, 배치 정규화 등 비활성화)
    model2 = torch.load('./data/models/gakzil_model_label3_84%.pt', map_location=torch.device('cpu'))
    model2.eval()  # 평가 모드로 전환 (드롭아웃, 배치 정규화 등 비활성화)
    model3 = torch.load('./data/models/hongban_label3_93.2%.pt', map_location=torch.device('cpu'))
    model3.eval()  # 평가 모드로 전환 (드롭아웃, 배치 정규화 등 비활성화)
    model4 = torch.load('./data/models/nongpo_model_label3_89.5.pt', map_location=torch.device('cpu'))
    model4.eval()  # 평가 모드로 전환 (드롭아웃, 배치 정규화 등 비활성화)
    model5 = torch.load('./data/models/pizy_model_92.6%.pt', map_location=torch.device('cpu'))
    model5.eval()  # 평가 모드로 전환 (드롭아웃, 배치 정규화 등 비활성화)
    model6 = torch.load('./data/models/talmo_model_93.48%.pt', map_location=torch.device('cpu'))
    model6.eval()  # 평가 모드로 전환 (드롭아웃, 배치 정규화 등 비활성화)

    return [model1, model2, model3, model4, model5, model6]

def load_image(image_path):

    transform = transforms.Compose([
        transforms.Lambda(lambda img: resize_with_aspect_ratio(img, target_size=240)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')  # 이미지를 RGB로 변환
    image = transform(image).unsqueeze(0)  # 배치 차원 추가 (1, 3, 224, 224)

    return image

# 이미지를 모델에 통과시켜 예측하는 함수
def predict_image(image_path):

    class_names = ['class1', 'class2', 'class3']

    models = load_models()
    model1 = models[0]
    model2 = models[1]
    model3 = models[2]
    model4 = models[3]
    model5 = models[4]
    model6 = models[5]

    # 장치 설정 (GPU 사용 가능 시 GPU로 이동)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model1 = model1.to(device)
    model2 = model2.to(device)
    model3 = model3.to(device)
    model4 = model4.to(device)
    model5 = model5.to(device)
    model6 = model6.to(device)


    image_tensor = load_image(image_path)  # 이미지를 전처리하여 텐서로 변환
    image_tensor = image_tensor.to(device)  # 모델과 동일한 장치로 이동 (CPU/GPU)

    with torch.no_grad():  # 예측 시 기울기 계산을 하지 않음
        outputs1 = model1(image_tensor)  # 모델의 예측값 (로짓)
        outputs2 = model2(image_tensor)  # 모델의 예측값 (로짓)
        outputs3 = model3(image_tensor)  # 모델의 예측값 (로짓)
        outputs4 = model4(image_tensor)  # 모델의 예측값 (로짓)
        outputs5 = model5(image_tensor)  # 모델의 예측값 (로짓)
        outputs6 = model6(image_tensor)  # 모델의 예측값 (로짓)
        _, preds1 = torch.max(outputs1, 1)  # 가장 높은 확률의 클래스 선택
        _, preds2 = torch.max(outputs2, 1)  # 가장 높은 확률의 클래스 선택
        _, preds3 = torch.max(outputs3, 1)  # 가장 높은 확률의 클래스 선택
        _, preds4 = torch.max(outputs4, 1)  # 가장 높은 확률의 클래스 선택
        _, preds5 = torch.max(outputs5, 1)  # 가장 높은 확률의 클래스 선택
        _, preds6 = torch.max(outputs6, 1)  # 가장 높은 확률의 클래스 선택

        probabilities1 = torch.nn.functional.softmax(outputs1, dim=1)
        probabilities2 = torch.nn.functional.softmax(outputs2, dim=1)
        probabilities3 = torch.nn.functional.softmax(outputs3, dim=1)
        probabilities4 = torch.nn.functional.softmax(outputs4, dim=1)
        probabilities5 = torch.nn.functional.softmax(outputs5, dim=1)
        probabilities6 = torch.nn.functional.softmax(outputs6, dim=1)

        # 각 이미지에 대한 클래스별 확률값 저장
        prob_values1 = probabilities1.cpu().numpy()[0]  # 확률값을 numpy 배열로 변환
        prob_values2 = probabilities2.cpu().numpy()[0]  # 확률값을 numpy 배열로 변환
        prob_values3 = probabilities3.cpu().numpy()[0]  # 확률값을 numpy 배열로 변환
        prob_values4 = probabilities4.cpu().numpy()[0]  # 확률값을 numpy 배열로 변환
        prob_values5 = probabilities5.cpu().numpy()[0]  # 확률값을 numpy 배열로 변환
        prob_values6 = probabilities6.cpu().numpy()[0]  # 확률값을 numpy 배열로 변환

        top_probability1 = prob_values1[0]  # 첫 번째 확률
        top_probability2 = prob_values2[0]  # 첫 번째 확률
        top_probability3 = prob_values3[0]  # 첫 번째 확률
        top_probability4 = prob_values4[0]  # 첫 번째 확률
        top_probability5 = prob_values5[0]  # 첫 번째 확률
        top_probability6 = prob_values6[0]  # 첫 번째 확률

        second_probability1 = prob_values1[1]  # 두 번째 확률
        second_probability2 = prob_values2[1]  # 두 번째 확률
        second_probability3 = prob_values3[1]  # 두 번째 확률
        second_probability4 = prob_values4[1]  # 두 번째 확률
        second_probability5 = prob_values5[1]  # 두 번째 확률
        second_probability6 = prob_values6[1]  # 두 번째 확률

        third_probability1 = prob_values1[2]  # 세 번째 확률
        third_probability2 = prob_values2[2]  # 세 번째 확률
        third_probability3 = prob_values3[2]  # 세 번째 확률
        third_probability4 = prob_values4[2]  # 세 번째 확률
        third_probability5 = prob_values5[2]  # 세 번째 확률
        third_probability6 = prob_values6[2]  # 세 번째 확률

        # 상위 확률 및 해당 클래스 찾기
        # top_two_indices1 = prob_values1.argsort()[-3:][::-1]  # 상위 2개의 인덱스 (내림차순)
        # top_two_indices2 = prob_values2.argsort()[-3:][::-1]  # 상위 2개의 인덱스 (내림차순)
        # top_two_indices3 = prob_values3.argsort()[-3:][::-1]  # 상위 2개의 인덱스 (내림차순)
        # top_two_indices4 = prob_values4.argsort()[-3:][::-1]  # 상위 2개의 인덱스 (내림차순)
        # top_two_indices5 = prob_values5.argsort()[-3:][::-1]  # 상위 2개의 인덱스 (내림차순)
        # top_two_indices6 = prob_values6.argsort()[-3:][::-1]  # 상위 2개의 인덱스 (내림차순)

        # top_class1 = class_names[top_two_indices1[0]]  # 첫 번째 클래스
        # top_class2 = class_names[top_two_indices2[0]]  # 첫 번째 클래스
        # top_class3 = class_names[top_two_indices3[0]]  # 첫 번째 클래스
        # top_class4 = class_names[top_two_indices4[0]]  # 첫 번째 클래스
        # top_class5 = class_names[top_two_indices5[0]]  # 첫 번째 클래스
        # top_class6 = class_names[top_two_indices6[0]]  # 첫 번째 클래스

        # top_probability1 = prob_values1[top_two_indices1[0]]  # 첫 번째 확률
        # top_probability2 = prob_values2[top_two_indices2[0]]  # 첫 번째 확률
        # top_probability3 = prob_values3[top_two_indices3[0]]  # 첫 번째 확률
        # top_probability4 = prob_values4[top_two_indices4[0]]  # 첫 번째 확률
        # top_probability5 = prob_values5[top_two_indices5[0]]  # 첫 번째 확률
        # top_probability6 = prob_values6[top_two_indices6[0]]  # 첫 번째 확률
        #
        # second_class1 = class_names[top_two_indices1[1]]  # 두 번째 클래스
        # second_class2 = class_names[top_two_indices2[1]]  # 두 번째 클래스
        # second_class3 = class_names[top_two_indices3[1]]  # 두 번째 클래스
        # second_class4 = class_names[top_two_indices4[1]]  # 두 번째 클래스
        # second_class5 = class_names[top_two_indices5[1]]  # 두 번째 클래스
        # second_class6 = class_names[top_two_indices6[1]]  # 두 번째 클래스
        #
        # second_probability1 = prob_values1[top_two_indices1[1]]  # 두 번째 확률
        # second_probability2 = prob_values2[top_two_indices2[1]]  # 두 번째 확률
        # second_probability3 = prob_values3[top_two_indices3[1]]  # 두 번째 확률
        # second_probability4 = prob_values4[top_two_indices4[1]]  # 두 번째 확률
        # second_probability5 = prob_values5[top_two_indices5[1]]  # 두 번째 확률
        # second_probability6 = prob_values6[top_two_indices6[1]]  # 두 번째 확률
        #
        # third_class1 = class_names[top_two_indices1[2]]  # 세 번째 클래스
        # third_class2 = class_names[top_two_indices2[2]]  # 세 번째 클래스
        # third_class3 = class_names[top_two_indices3[2]]  # 세 번째 클래스
        # third_class4 = class_names[top_two_indices4[2]]  # 세 번째 클래스
        # third_class5 = class_names[top_two_indices5[2]]  # 세 번째 클래스
        # third_class6 = class_names[top_two_indices6[2]]  # 세 번째 클래스
        #
        # third_probability1 = prob_values1[top_two_indices1[2]]  # 세 번째 확률
        # third_probability2 = prob_values2[top_two_indices2[2]]  # 세 번째 확률
        # third_probability3 = prob_values3[top_two_indices3[2]]  # 세 번째 확률
        # third_probability4 = prob_values4[top_two_indices4[2]]  # 세 번째 확률
        # third_probability5 = prob_values5[top_two_indices5[2]]  # 세 번째 확률
        # third_probability6 = prob_values6[top_two_indices6[2]]  # 세 번째 확률


    # return [[preds1.item(), top_class1, top_probability1, second_class1, second_probability1, third_class1, third_probability1],
    #         [preds2.item(), top_class2, top_probability2, second_class2, second_probability2, third_class2, third_probability2],
    #         [preds3.item(), top_class3, top_probability3, second_class3, second_probability3, third_class3, third_probability3],
    #         [preds4.item(), top_class4, top_probability4, second_class4, second_probability4, third_class4, third_probability4],
    #         [preds5.item(), top_class5, top_probability5, second_class5, second_probability5, third_class5, third_probability5],
    #         [preds6.item(), top_class6, top_probability6, second_class6, second_probability6, third_class6, third_probability6]]  # 예측된 클래스 반환

    return [[preds1.item(), top_probability1, second_probability1, third_probability1],
            [preds2.item(), top_probability2, second_probability2, third_probability2],
            [preds3.item(), top_probability3, second_probability3, third_probability3],
            [preds4.item(), top_probability4, second_probability4, third_probability4],
            [preds5.item(), top_probability5, second_probability5, third_probability5],
            [preds6.item(), top_probability6, second_probability6, third_probability6]]  # 예측된 클래스 반환


def request_chat_completion(prompt):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "당신은 두피 전문가 입니다."},
            {"role": "user", "content": prompt}
        ],
        stream=True
    )
    return response


def generate_prompt(scalp_type):
    prompt = f"""
{scalp_type}의 원인과 특징 그리고 관리방안을 생성해주세요. 
각 타입별 원인, 특징을 각각 1문장으로 말해주고, 관리방안은 3줄 정도로 자세하게 적어주세요. 

반드시 키워드를 포함해야 합니다.
이모지를 사용하세요.
글씨 크기는 일정하게 해주세요.
---
두피 타입: {scalp_type}
---
""".strip()
    return prompt


def print_streaming_response(response):
    message = ""
    placeholder = st.empty()
    for chunk in response:
        delta = chunk.choices[0].delta
        if delta.content:
            message += delta.content
            placeholder.markdown(message + "▌")
    placeholder.markdown(message)


@st.cache_data
def load_data(variety):
    if variety == "shampoo":
        df = pd.read_csv("./data/crowlings/shampoo_ingredient_data2_add_type.csv", encoding="utf-8-sig")
    elif variety == "rinse":
        df = pd.read_csv("./data/crowlings/rinse_ingredient_data2_add_type.csv", encoding="utf-8-sig")
    elif variety == "bar":
        df = pd.read_csv("./data/crowlings/bar_ingredient_data2_add_type.csv", encoding="utf-8-sig")
    elif variety == "hairoil":
        df = pd.read_csv("./data/crowlings/hairoil_ingredient_data2_add_type.csv", encoding="utf-8-sig")
    elif variety == "hairwater":
        df = pd.read_csv("./data/crowlings/hairwater_ingredient_data2_add_type.csv", encoding="utf-8-sig")
    elif variety == "scaler":
        df = pd.read_csv("./data/crowlings/scaler_ingredient_data2_add_type.csv", encoding="utf-8-sig")
    elif variety == "tonic":
        df = pd.read_csv("./data/crowlings/tonic_ingredient_data2_add_type.csv", encoding="utf-8-sig")
    return df

def product_recommend(df):
    # 클래스 이름 정의
    class_names = ["👻 양호", "💧 경증", "😈 중증"]  # 클래스

    bidum_state = st.session_state.scalp[0]["bidum_state"]
    gakzil_state = st.session_state.scalp[0]["gakzil_state"]
    hongban_state = st.session_state.scalp[0]["hongban_state"]
    nongpo_state = st.session_state.scalp[0]["nongpo_state"]
    pizy_state = st.session_state.scalp[0]["pizy_state"]
    talmo_state = st.session_state.scalp[0]["talmo_state"]

    print(f"비듬상태 : {bidum_state}, 각질상태 : {gakzil_state}, 홍반상태 : {hongban_state}, 농포상태 : {nongpo_state}, 피지상태 : {pizy_state}, 탈모상태 : {talmo_state}")

    recommend_type_product = {
        "건성" : [],
        "지성" : [],
        "지루성" : [],
        "비듬성" : [],
        "탈모성" : []
    }

    result_type = []

    # 6개 증상이 모두 경증일 때 - > 지성, 건성 중에 하나로 가자
    if bidum_state == class_names[1] and gakzil_state == class_names[1] and hongban_state == class_names[1] and nongpo_state == class_names[1] and pizy_state == class_names[1] and talmo_state == class_names[1]:
        if "".join(st.session_state.scalp[0]["type"]) == "(건성)두피에 건조함이나 당김을 느낍니다.":
            data = []
            for i in range(len(df)):
                type_line = str(df.iloc[i]["type"])
                if type_line:
                    row = type_line.split(",")
                    for j in range(len(row)):
                        if row[j] == "건성":
                            data.append([df.iloc[i]["product_link"], df.iloc[i]["img_link"], df.iloc[i]["brand_name"],
                                         df.iloc[i]["product_name"],
                                         df.iloc[i]["star"], df.iloc[i]["review_count"], df.iloc[i]["ingredient"]])
                            break
            recommend_type_product["건성"] = data
            result_type.append("건성")
            print("6개 증상이 모두 경증일 때 - > 건성")
        elif "".join(st.session_state.scalp[0]["type"]) == "(지성)머리를 감은지 하루 이내에 두피가 기름집니다.":
            data = []
            for i in range(len(df)):
                type_line = str(df.iloc[i]["type"])
                if type_line:
                    row = type_line.split(",")
                    for j in range(len(row)):
                        if row[j] == "지성":
                            data.append([df.iloc[i]["product_link"], df.iloc[i]["img_link"], df.iloc[i]["brand_name"],
                                         df.iloc[i]["product_name"],
                                         df.iloc[i]["star"], df.iloc[i]["review_count"], df.iloc[i]["ingredient"]])
                            break
            recommend_type_product["지성"] = data
            result_type.append("지성")
            print("6개 증상이 모두 경증일 때 - > 지성")

    # 6개의 증상 중에 하나만 중증 일때 -> 각질 : 건성
    elif gakzil_state == class_names[2] and bidum_state != class_names[2] and hongban_state != class_names[2] and nongpo_state != class_names[2] and pizy_state != class_names[2] and talmo_state != class_names[2]:
        data = []
        for i in range(len(df)):
            type_line = str(df.iloc[i]["type"])
            if type_line:
                row = type_line.split(",")
                for j in range(len(row)):
                    if row[j] == "건성":
                        data.append([df.iloc[i]["product_link"], df.iloc[i]["img_link"], df.iloc[i]["brand_name"],
                                     df.iloc[i]["product_name"],
                                     df.iloc[i]["star"], df.iloc[i]["review_count"], df.iloc[i]["ingredient"]])
                        break
        recommend_type_product["건성"] = data
        result_type.append("건성")
        print("6개의 증상 중에 하나만 중증 일때 -> 각질 : 건성")

    # 6개의 증상 중에 하나만 중증 일때 -> 피지 : 지성
    elif pizy_state == class_names[2] and bidum_state != class_names[2] and hongban_state != class_names[2] and nongpo_state != class_names[2] and gakzil_state != class_names[2] and talmo_state != class_names[2]:
        data = []
        for i in range(len(df)):
            type_line = str(df.iloc[i]["type"])
            if type_line:
                row = type_line.split(",")
                for j in range(len(row)):
                    if row[j] == "지성":
                        data.append([df.iloc[i]["product_link"], df.iloc[i]["img_link"], df.iloc[i]["brand_name"],
                                     df.iloc[i]["product_name"],
                                     df.iloc[i]["star"], df.iloc[i]["review_count"], df.iloc[i]["ingredient"]])
                        break
        recommend_type_product["지성"] = data
        result_type.append("지성")
        print("6개의 증상 중에 하나만 중증 일때 -> 피지 : 지성")

    # 6개의 증상 중에 하나만 중증 일때 -> 모낭사이홍반 : 설문(지성, 건성),  상반될 시 모델링
    elif hongban_state == class_names[2] and bidum_state != class_names[2] and pizy_state != class_names[2] and nongpo_state != class_names[2] and gakzil_state != class_names[2] and talmo_state != class_names[2]:
        print(st.session_state.scalp[0]["type"])
        if "".join(st.session_state.scalp[0]["type"]) == "(건성)두피에 건조함이나 당김을 느낍니다.":
            data = []
            for i in range(len(df)):
                type_line = str(df.iloc[i]["type"])
                if type_line:
                    row = type_line.split(",")
                    for j in range(len(row)):
                        if row[j] == "건성":
                            data.append(
                                [df.iloc[i]["product_link"], df.iloc[i]["img_link"], df.iloc[i]["brand_name"],
                                 df.iloc[i]["product_name"],
                                 df.iloc[i]["star"], df.iloc[i]["review_count"], df.iloc[i]["ingredient"]])
                            break
            recommend_type_product["건성"] = data
            result_type.append("건성")
            print("6개의 증상 중에 하나만 중증 일때 -> 모낭사이홍반 : 설문(건성)")

        elif "".join(st.session_state.scalp[0]["type"]) == "(지성)머리를 감은지 하루 이내에 두피가 기름집니다.":
            data = []
            for i in range(len(df)):
                type_line = str(df.iloc[i]["type"])
                if type_line:
                    row = type_line.split(",")
                    for j in range(len(row)):
                        if row[j] == "지성":
                            data.append(
                                [df.iloc[i]["product_link"], df.iloc[i]["img_link"], df.iloc[i]["brand_name"],
                                 df.iloc[i]["product_name"],
                                 df.iloc[i]["star"], df.iloc[i]["review_count"], df.iloc[i]["ingredient"]])
                            break
            recommend_type_product["지성"] = data
            result_type.append("지성")
            print("6개의 증상 중에 하나만 중증 일때 -> 모낭사이홍반 : 설문(지성)")

    # 6개의 증상 중에 하나만 중증 일때 -> 모낭홍반농포 : 지루성
    elif nongpo_state == class_names[2] and bidum_state != class_names[2] and hongban_state != class_names[2] and pizy_state != class_names[2] and gakzil_state != class_names[2] and talmo_state != class_names[2]:
        data = []
        for i in range(len(df)):
            type_line = str(df.iloc[i]["type"])
            if type_line:
                row = type_line.split(",")
                for j in range(len(row)):
                    if row[j] == "지루성":
                        data.append([df.iloc[i]["product_link"], df.iloc[i]["img_link"], df.iloc[i]["brand_name"],
                                     df.iloc[i]["product_name"],
                                     df.iloc[i]["star"], df.iloc[i]["review_count"], df.iloc[i]["ingredient"]])
                        break

        recommend_type_product["지루성"] = data
        result_type.append("지루성")
        print("6개의 증상 중에 하나만 중증 일때 -> 모낭홍반농포 : 지루성")

    # 6개의 증상 중에 하나만 중증 일때 -> 비듬 : 비듬성
    elif bidum_state == class_names[2] and nongpo_state != class_names[2] and hongban_state != class_names[2] and pizy_state != class_names[2] and gakzil_state != class_names[2] and talmo_state != class_names[2]:
        data = []
        for i in range(len(df)):
            type_line = str(df.iloc[i]["type"])
            if type_line:
                row = type_line.split(",")
                for j in range(len(row)):
                    if row[j] == "비듬":
                        data.append([df.iloc[i]["product_link"], df.iloc[i]["img_link"], df.iloc[i]["brand_name"],
                                     df.iloc[i]["product_name"],
                                     df.iloc[i]["star"], df.iloc[i]["review_count"], df.iloc[i]["ingredient"]])
                        break
        recommend_type_product["비듬성"] = data
        result_type.append("비듬성")
        print("6개의 증상 중에 하나만 중증 일때 -> 비듬 : 비듬성")

    # 6개의 증상 중에 하나만 중증 일때 -> 탈모 : 탈모성
    elif talmo_state == class_names[2] and nongpo_state != class_names[2] and hongban_state != class_names[2] and pizy_state != class_names[2] and gakzil_state != class_names[2] and bidum_state != class_names[2]:
        data = []
        for i in range(len(df)):
            type_line = str(df.iloc[i]["type"])
            if type_line:
                row = type_line.split(",")
                for j in range(len(row)):
                    if row[j] == "탈모":
                        data.append([df.iloc[i]["product_link"], df.iloc[i]["img_link"], df.iloc[i]["brand_name"],
                                     df.iloc[i]["product_name"],
                                     df.iloc[i]["star"], df.iloc[i]["review_count"], df.iloc[i]["ingredient"]])
                        break
        recommend_type_product["탈모성"] = data
        result_type.append("탈모성")
        print("6개의 증상 중에 하나만 중증 일때 -> 탈모 : 탈모성")

    # 각질 : 중증, 피지 : 중증, 홍반 : 중증, 농포 : 중증, 비듬 : 중증, 탈모 : 중증 (가능성이 없음)
    # 각질과 비듬 : 건성 추천, 피지와 홍반 : 지성 추천, 피지와 농포 : 지루성 추천, 각질과 비듬 : 비듬성 추천, 탈모 : 탈모 추천

    # 각질 : 중증, 피지 : 양호, 홍반 : 중증, 농포 : 중증, 비듬 : 중증, 탈모 : 중증
    # 각질과 비듬 : 건성 추천, 피지와 농포 : 지루성 추천, 각질과 비듬 : 비듬성 추천, 탈모 : 탈모 추천

    # 각질 : 중증, 피지 : 양호, 홍반 : 중증, 농포 : 중증, 비듬 : 경증, 탈모 : 중증
    # 각질과 비듬 : 건성 추천, 각질과 비듬 : 비듬성 추천, 탈모 : 탈모 추천

    # 각질 : 중증, 피지 : 양호, 홍반 : 중증, 농포 : 중증, 비듬 : 양호, 탈모 : 중증
    # 각질과 비듬 : 건성 추천?, 피지와 농포 : 지루성 추천?? 각질과 비듬 : 비듬성 추천??, 탈모 : 탈모 추천


    # 가장 높은 중증도가 2개 또는 그 이상이면서 두피 증상 표 이외의 경우일 때
    # 모든 두피 증상과 관련된 제품 모두 추천
    else:

        if talmo_state == class_names[2] or nongpo_state == class_names[2] or hongban_state == class_names[2] or pizy_state == class_names[2] or gakzil_state == class_names[2] or bidum_state == class_names[2]:
            if gakzil_state == class_names[2] and (hongban_state == class_names[2] and "".join(st.session_state.scalp[0]["type"]) == "(건성)두피에 건조함이나 당김을 느낍니다."):
                data = []
                for i in range(len(df)):
                    type_line = str(df.iloc[i]["type"])
                    if type_line:
                        row = type_line.split(",")
                        for j in range(len(row)):
                            if row[j] == "건성":
                                data.append([df.iloc[i]["product_link"], df.iloc[i]["img_link"], df.iloc[i]["brand_name"],
                                             df.iloc[i]["product_name"],
                                             df.iloc[i]["star"], df.iloc[i]["review_count"], df.iloc[i]["ingredient"]])
                                break
                recommend_type_product["건성"] = data
                result_type.append("건성")
                print("각질과 홍반이 중증이어서 건성으로 판단 한 경우")

            elif pizy_state == class_names[2] and (hongban_state == class_names[2] and "".join(st.session_state.scalp[0]["type"]) == "(지성)머리를 감은지 하루 이내에 두피가 기름집니다."):
                data = []
                for i in range(len(df)):
                    type_line = str(df.iloc[i]["type"])
                    if type_line:
                        row = type_line.split(",")
                        for j in range(len(row)):
                            if row[j] == "지성":
                                data.append([df.iloc[i]["product_link"], df.iloc[i]["img_link"], df.iloc[i]["brand_name"],
                                             df.iloc[i]["product_name"],
                                             df.iloc[i]["star"], df.iloc[i]["review_count"], df.iloc[i]["ingredient"]])
                                break
                recommend_type_product["지성"] = data
                result_type.append("지성")
                print("피지와 홍반이 중증이어서 지성으로 판단한 경우")
            else:
                if gakzil_state == class_names[2]:
                    data = []
                    for i in range(len(df)):
                        type_line = str(df.iloc[i]["type"])
                        if type_line:
                            row = type_line.split(",")
                            for j in range(len(row)):
                                if row[j] == "건성":
                                    data.append(
                                        [df.iloc[i]["product_link"], df.iloc[i]["img_link"], df.iloc[i]["brand_name"],
                                         df.iloc[i]["product_name"],
                                         df.iloc[i]["star"], df.iloc[i]["review_count"], df.iloc[i]["ingredient"]])
                                    break
                    recommend_type_product["건성"] = data
                    result_type.append("건성")
                    print("각질이 중증이어서 건성으로 판단 한 경우")

                elif pizy_state == class_names[2]:
                    data = []
                    for i in range(len(df)):
                        type_line = str(df.iloc[i]["type"])
                        if type_line:
                            row = type_line.split(",")
                            for j in range(len(row)):
                                if row[j] == "지성":
                                    data.append(
                                        [df.iloc[i]["product_link"], df.iloc[i]["img_link"], df.iloc[i]["brand_name"],
                                         df.iloc[i]["product_name"],
                                         df.iloc[i]["star"], df.iloc[i]["review_count"], df.iloc[i]["ingredient"]])
                                    break
                    recommend_type_product["지성"] = data
                    result_type.append("지성")
                    print("피지가 중증이어서 지성으로 판단한 경우")
                    
                elif hongban_state == class_names[2] and "".join(st.session_state.scalp[0]["type"]) == "(건성)두피에 건조함이나 당김을 느낍니다.":
                    data = []
                    for i in range(len(df)):
                        type_line = str(df.iloc[i]["type"])
                        if type_line:
                            row = type_line.split(",")
                            for j in range(len(row)):
                                if row[j] == "건성":
                                    data.append(
                                        [df.iloc[i]["product_link"], df.iloc[i]["img_link"], df.iloc[i]["brand_name"],
                                         df.iloc[i]["product_name"],
                                         df.iloc[i]["star"], df.iloc[i]["review_count"], df.iloc[i]["ingredient"]])
                                    break
                    recommend_type_product["건성"] = data
                    result_type.append("건성")
                    print("홍반이 중증이고 두피타입이 건성이어서 건성으로 판단 한 경우")

                elif hongban_state == class_names[2] and "".join(st.session_state.scalp[0]["type"]) == "(지성)머리를 감은지 하루 이내에 두피가 기름집니다.":
                    data = []
                    for i in range(len(df)):
                        type_line = str(df.iloc[i]["type"])
                        if type_line:
                            row = type_line.split(",")
                            for j in range(len(row)):
                                if row[j] == "지성":
                                    data.append(
                                        [df.iloc[i]["product_link"], df.iloc[i]["img_link"], df.iloc[i]["brand_name"],
                                         df.iloc[i]["product_name"],
                                         df.iloc[i]["star"], df.iloc[i]["review_count"], df.iloc[i]["ingredient"]])
                                    break
                    recommend_type_product["지성"] = data
                    result_type.append("지성")
                    print("홍반이 중증이고 두피타입이 지성이어서 지성으로 판단한 경우")


            if nongpo_state == class_names[2]:
                data = []
                for i in range(len(df)):
                    type_line = str(df.iloc[i]["type"])
                    if type_line:
                        row = type_line.split(",")
                        for j in range(len(row)):
                            if row[j] == "지루성":
                                data.append([df.iloc[i]["product_link"], df.iloc[i]["img_link"], df.iloc[i]["brand_name"],
                                             df.iloc[i]["product_name"],
                                             df.iloc[i]["star"], df.iloc[i]["review_count"], df.iloc[i]["ingredient"]])
                                break
                recommend_type_product["지루성"] = data
                result_type.append("지루성")
                print("농포가 중증이어서 지루성으로 판단한 경우")

            if bidum_state == class_names[2]:
                data = []
                for i in range(len(df)):
                    type_line = str(df.iloc[i]["type"])
                    if type_line:
                        row = type_line.split(",")
                        for j in range(len(row)):
                            if row[j] == "비듬":
                                data.append([df.iloc[i]["product_link"], df.iloc[i]["img_link"], df.iloc[i]["brand_name"],
                                             df.iloc[i]["product_name"],
                                             df.iloc[i]["star"], df.iloc[i]["review_count"], df.iloc[i]["ingredient"]])
                                break
                recommend_type_product["비듬성"] = data
                result_type.append("비듬성")
                print("비듬이 중증이어서 비듬성으로 판단한 경우 ")

            if talmo_state == class_names[2]:
                data = []
                for i in range(len(df)):
                    type_line = str(df.iloc[i]["type"])
                    if type_line:
                        row = type_line.split(",")
                        for j in range(len(row)):
                            if row[j] == "탈모":
                                data.append([df.iloc[i]["product_link"], df.iloc[i]["img_link"], df.iloc[i]["brand_name"],
                                             df.iloc[i]["product_name"],
                                             df.iloc[i]["star"], df.iloc[i]["review_count"], df.iloc[i]["ingredient"]])
                                break
                recommend_type_product["탈모성"] = data
                result_type.append("탈모성")
                print("탈모가 중증이어서 탈모성으로 판단한 경우")

        elif talmo_state == class_names[1] or nongpo_state == class_names[1] or hongban_state == class_names[1] or pizy_state == class_names[1] or gakzil_state == class_names[1] or bidum_state == class_names[1]:
            if gakzil_state == class_names[1] and (hongban_state == class_names[1] and "".join(st.session_state.scalp[0]["type"]) == "(건성)두피에 건조함이나 당김을 느낍니다."):
                data = []
                for i in range(len(df)):
                    type_line = str(df.iloc[i]["type"])
                    if type_line:
                        row = type_line.split(",")
                        for j in range(len(row)):
                            if row[j] == "건성":
                                data.append(
                                    [df.iloc[i]["product_link"], df.iloc[i]["img_link"], df.iloc[i]["brand_name"],
                                     df.iloc[i]["product_name"],
                                     df.iloc[i]["star"], df.iloc[i]["review_count"], df.iloc[i]["ingredient"]])
                                break
                recommend_type_product["건성"] = data
                result_type.append("건성")
                print("각질과 홍반이 경증이어서 건성으로 판단한 경우")

            elif pizy_state == class_names[1] and (hongban_state == class_names[1] and "".join(st.session_state.scalp[0]["type"]) == "(지성)머리를 감은지 하루 이내에 두피가 기름집니다."):
                data = []
                for i in range(len(df)):
                    type_line = str(df.iloc[i]["type"])
                    if type_line:
                        row = type_line.split(",")
                        for j in range(len(row)):
                            if row[j] == "지성":
                                data.append(
                                    [df.iloc[i]["product_link"], df.iloc[i]["img_link"], df.iloc[i]["brand_name"],
                                     df.iloc[i]["product_name"],
                                     df.iloc[i]["star"], df.iloc[i]["review_count"], df.iloc[i]["ingredient"]])
                                break
                recommend_type_product["지성"] = data
                result_type.append("지성")
                print("피지와 홍반이 경증이어서 지성으로 판단한 경우")

            else:
                if gakzil_state == class_names[1]:
                    data = []
                    for i in range(len(df)):
                        type_line = str(df.iloc[i]["type"])
                        if type_line:
                            row = type_line.split(",")
                            for j in range(len(row)):
                                if row[j] == "건성":
                                    data.append(
                                        [df.iloc[i]["product_link"], df.iloc[i]["img_link"], df.iloc[i]["brand_name"],
                                         df.iloc[i]["product_name"],
                                         df.iloc[i]["star"], df.iloc[i]["review_count"], df.iloc[i]["ingredient"]])
                                    break
                    recommend_type_product["건성"] = data
                    result_type.append("건성")
                    print("각질이 경증이어서 건성으로 판단 한 경우")

                elif pizy_state == class_names[1]:
                    data = []
                    for i in range(len(df)):
                        type_line = str(df.iloc[i]["type"])
                        if type_line:
                            row = type_line.split(",")
                            for j in range(len(row)):
                                if row[j] == "지성":
                                    data.append(
                                        [df.iloc[i]["product_link"], df.iloc[i]["img_link"], df.iloc[i]["brand_name"],
                                         df.iloc[i]["product_name"],
                                         df.iloc[i]["star"], df.iloc[i]["review_count"], df.iloc[i]["ingredient"]])
                                    break
                    recommend_type_product["지성"] = data
                    result_type.append("지성")
                    print("피지가 경증이어서 지성으로 판단한 경우")

                elif hongban_state == class_names[1] and "".join(st.session_state.scalp[0]["type"]) == "(건성)두피에 건조함이나 당김을 느낍니다.":
                    data = []
                    for i in range(len(df)):
                        type_line = str(df.iloc[i]["type"])
                        if type_line:
                            row = type_line.split(",")
                            for j in range(len(row)):
                                if row[j] == "건성":
                                    data.append(
                                        [df.iloc[i]["product_link"], df.iloc[i]["img_link"], df.iloc[i]["brand_name"],
                                         df.iloc[i]["product_name"],
                                         df.iloc[i]["star"], df.iloc[i]["review_count"], df.iloc[i]["ingredient"]])
                                    break
                    recommend_type_product["건성"] = data
                    result_type.append("건성")
                    print("홍반이 경증이고 두피타입이 건성이어서 건성으로 판단 한 경우")

                elif hongban_state == class_names[1] and "".join(st.session_state.scalp[0]["type"]) == "(지성)머리를 감은지 하루 이내에 두피가 기름집니다.":
                    data = []
                    for i in range(len(df)):
                        type_line = str(df.iloc[i]["type"])
                        if type_line:
                            row = type_line.split(",")
                            for j in range(len(row)):
                                if row[j] == "지성":
                                    data.append(
                                        [df.iloc[i]["product_link"], df.iloc[i]["img_link"], df.iloc[i]["brand_name"],
                                         df.iloc[i]["product_name"],
                                         df.iloc[i]["star"], df.iloc[i]["review_count"], df.iloc[i]["ingredient"]])
                                    break
                    recommend_type_product["지성"] = data
                    result_type.append("지성")
                    print("홍반이 경증이고 두피타입이 지성이어서 지성으로 판단한 경우")

            if nongpo_state == class_names[1]:
                data = []
                for i in range(len(df)):
                    type_line = str(df.iloc[i]["type"])
                    if type_line:
                        row = type_line.split(",")
                        for j in range(len(row)):
                            if row[j] == "지루성":
                                data.append(
                                    [df.iloc[i]["product_link"], df.iloc[i]["img_link"], df.iloc[i]["brand_name"],
                                     df.iloc[i]["product_name"],
                                     df.iloc[i]["star"], df.iloc[i]["review_count"], df.iloc[i]["ingredient"]])
                                break
                recommend_type_product["지루성"] = data
                result_type.append("지루성")
                print("농포가 경증이어서 지루성으로 판단한 경우")

            if bidum_state == class_names[1]:
                data = []
                for i in range(len(df)):
                    type_line = str(df.iloc[i]["type"])
                    if type_line:
                        row = type_line.split(",")
                        for j in range(len(row)):
                            if row[j] == "비듬":
                                data.append(
                                    [df.iloc[i]["product_link"], df.iloc[i]["img_link"], df.iloc[i]["brand_name"],
                                     df.iloc[i]["product_name"],
                                     df.iloc[i]["star"], df.iloc[i]["review_count"], df.iloc[i]["ingredient"]])
                                break
                recommend_type_product["비듬성"] = data
                result_type.append("비듬성")
                print("비듬이 경증이어서 비듬성으로 판단한 경우")

            if talmo_state == class_names[1]:
                data = []
                for i in range(len(df)):
                    type_line = str(df.iloc[i]["type"])
                    if type_line:
                        row = type_line.split(",")
                        for j in range(len(row)):
                            if row[j] == "탈모":
                                data.append(
                                    [df.iloc[i]["product_link"], df.iloc[i]["img_link"], df.iloc[i]["brand_name"],
                                     df.iloc[i]["product_name"],
                                     df.iloc[i]["star"], df.iloc[i]["review_count"], df.iloc[i]["ingredient"]])
                                break
                recommend_type_product["탈모성"] = data
                result_type.append("탈모성")
                print("탈모가 경증이어서 탈모성으로 판단한 경우")

    st.session_state.scalp[0]["result_type"] = result_type

    return recommend_type_product

def product_find(result, str_variety):

    data = []
    view_result = []
    if result["건성"] != []:
        tmp.append("건성")
        data += result["건성"]

    if result["지성"] != []:
        tmp.append("지성")
        data += result["지성"]

    if result["지루성"] != []:
        tmp.append("지루성")
        data += result["지루성"]

    if result["비듬성"] != []:
        tmp.append("비듬성")
        data += result["비듬성"]

    if result["탈모성"] != []:
        tmp.append("탈모성")
        data += result["탈모성"]


    # 정렬
    if data:
        data = sorted(data, key=lambda x: (x[4], x[5]), reverse=True)

        if data:
            # 중복 제거
            tmp_list = []
            tmp2 = []
            for k in range(len(data)):
                if data[k][2] not in tmp_list:
                    tmp_list.append(data[k][2])
                    tmp2.append(data[k])
            if tmp2:
                view_result = sorted(tmp2, key=lambda x: (x[4], x[5]), reverse=True)[:3]

    st.markdown(f"※ **⚜️{",".join(st.session_state.scalp[0]["result_type"])}** 두피타입에 맞는 **🧴 {str_variety}** 제품을 **🖐️{len(tmp2)}개 중 3개** 추천해드릴게요.")

    return view_result

def product_view(result):

    if len(result) == 3:
        tmp = []
        cols = st.columns(3)
        for l in range(3):
            with cols[l]:
                product_link = result[l][0]
                img_link = result[l][1]
                brand_name = result[l][2]
                product_name = result[l][3]
                if brand_name in tmp:
                    continue
                else:
                    tmp.append(brand_name)

                with st.expander(label=f"**{l + 1}. {product_name}**", expanded=True):
                    # st.markdown(f"{l+1}. {product_name}")
                    st.markdown(f'''
                        <a href="{product_link}" target="_blank">
                            <img src="{img_link}" alt="image" style="width: 200px;">
                        </a>
                        ''', unsafe_allow_html=True)
    elif len(result) == 2:
        tmp = []
        cols = st.columns(3)
        for l in range(2):
            with cols[l]:
                product_link = result[l][0]
                img_link = result[l][1]
                brand_name = result[l][2]
                product_name = result[l][3]
                if brand_name in tmp:
                    continue
                else:
                    tmp.append(brand_name)

                with st.expander(label=f"**{l + 1}. {product_name}**", expanded=True):
                    # st.markdown(f"{l+1}. {product_name}")
                    st.markdown(f'''
                        <a href="{product_link}" target="_blank">
                            <img src="{img_link}" alt="image" style="width: 200px;">
                        </a>
                        ''', unsafe_allow_html=True)
        with cols[2]:
            st.write("")
    elif len(result) == 1:
        tmp = []
        cols = st.columns(2)
        for l in range(1):
            with cols[l]:
                product_link = result[l][0]
                img_link = result[l][1]
                brand_name = result[l][2]
                product_name = result[l][3]
                if brand_name in tmp:
                    continue
                else:
                    tmp.append(brand_name)

                with st.expander(label=f"**{l + 1}. {product_name}**", expanded=True):
                    # st.markdown(f"{l+1}. {product_name}")
                    st.markdown(f'''
                        <a href="{product_link}" target="_blank">
                            <img src="{img_link}" alt="image" style="width: 200px;">
                        </a>
                        ''', unsafe_allow_html=True)
        with cols[1]:
            st.write("")
        with cols[2]:
            st.write("")

def product_description(result):

    df = pd.read_csv("./data/crowlings/major_ingredient_per_type.csv", encoding="EUC-KR")
    type_list = st.session_state.scalp[0]["result_type"]

    with st.expander(label=f"제품 설명서", expanded=True):
        for i in range(len(result)):
            type_line = str(result[i][6])
            data = []
            cnt = 0
            if type_line:
                row = type_line.split(",")
                for j in range(len(row)):
                    ingredient = row[j]
                    for k in type_list:
                        for l in range(len(df)):
                            major_ingredient = str(df.iloc[l][k])
                            str_len = len(ingredient)
                            sub_len = len(major_ingredient)
                            for q in range(str_len - sub_len + 1):
                                if ingredient[q:q + sub_len] == major_ingredient:
                                    if major_ingredient not in data:
                                        data.append(major_ingredient)
                                        cnt += 1
                                        break
                        if cnt > 2:
                            break
                    if cnt > 2:
                        break

                # st.markdown(f"{i+1}. **🧴 {result[i][3]}**은 **🍨{",".join(data)}** 성분 등이 들어 있어 **⚜️{symptom}** 두피 타입과 **{type}** 증상에 좋습니다.")
                st.markdown(f'''
                        {i+1}. <b>{result[i][2]}</b>은 <b>🍨{",".join(data)}</b> 성분 등이 들어 있어 당신의 두피 타입과 증상에 좋습니다.
                        ''', unsafe_allow_html=True)

# User-Agent 정보를 가져와서 세션에 저장하는 함수
# JavaScript에서 받은 User-Agent 처리
def handle_user_agent(data):
    if "user_agent" in data:
        st.session_state._user_agent = data["user_agent"]

# 기기 판별 함수
def detect_device(user_agent):
    if any(mobile in user_agent.lower() for mobile in ["iphone", "android", "ipad", "mobile"]):
        return "모바일"
    return "PC"


if "scalp" not in st.session_state:
    st.session_state.scalp = initial_scalp


if 'page' not in st.session_state:
    st.session_state.page = 0

if 'upload' not in st.session_state:
    st.session_state.upload = initial_upload


if 'survey' not in st.session_state:
    st.session_state.survey = 0

def next_page():
    st.session_state.page += 1

def prev_page():
    st.session_state.page = max(0, st.session_state.page - 1)

def prev_main_page():
    st.session_state.upload["session"] = 0
    st.session_state.page = max(0, st.session_state.page - 1)

def home_page():
    # st.experimental_js("location.reload()")
    # st.experimental_rerun()
    st.session_state.page = 0

if st.session_state.page == 0:
    ############################ 1. 사용자 두피 이미지 업로드  ############################
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**🔥 이미지 업로드 시 참고사항**")
        with st.expander(label="※ 클릭시 이미지 확장/삭제", expanded=True):
            st.image("./data/notice.jpg", use_column_width=True)


    with col2:
        st.markdown("**🔥 사용자 두피 이미지 업로드**")
        with st.expander(label="※ 클릭시 이미지 확장/삭제", expanded=True):
            # JavaScript와 HTML을 통해 User-Agent를 가져오는 코드
            user_agent = components.html(
                """
                <script>
                    // User-Agent를 가져와 Streamlit의 iframe에 전달
                    const userAgent = navigator.userAgent;
                    const message = {user_agent: userAgent};
                    window.parent.postMessage(message, "*");
                </script>
                """,
                height=0,
            )

            # Streamlit에서 postMessage 이벤트 처리
            if "_user_agent" not in st.session_state:
                st.session_state._user_agent = ""

            # User-Agent를 통해 기기 판별
            device_type = detect_device(st.session_state._user_agent)            

            if user_agent:
                # 모바일 기기 여부를 확인하는 단어들이 User-Agent에 포함되었는지 확인
                if device_type == "모바일":

                    uploaded_file = st.camera_input("사진을 찍어 주세요!")

                    st.write("")
                    st.write("")

                    # 저장할 경로 설정
                    SAVE_FOLDER = './data/uploaded_images/'

                    # 폴더가 존재하지 않으면 생성
                    if not os.path.exists(SAVE_FOLDER):
                        os.makedirs(SAVE_FOLDER)

                    # 파일이 업로드된 경우 처리
                    if uploaded_file is not None:
                        # 업로드된 파일을 PIL 이미지로 열기
                        image = Image.open(uploaded_file)

                        # 파일 이름을 가져옴
                        file_name = uploaded_file.name

                        # 저장할 경로 생성
                        file_path = os.path.join(SAVE_FOLDER, file_name)

                        # 이미지 파일을 지정한 경로에 저장
                        image.save(file_path)
                        st.session_state.upload["session"] = 1
                        st.session_state.upload["filepath"] = file_path
                        st.session_state.upload["filename"] = file_name
                        st.text("이미지가 성공적으로 업로드되었습니다.😍")
                        st.write("")


                    ############################ 2. 사용자 두피 이미지 결과  ############################
                    # if uploaded_file is not None and st.session_state.upload["session"] == 1:
                    #     st.markdown("**🔥 사용자 두피 이미지 보기**")
                    #     with st.expander(label="※ 사용자 두피 이미지", expanded=True):
                    #         st.image(image, caption='Uploaded Image.', use_column_width=True)
                    #         # st.write("이미지가 성공적으로 업로드 되었습니다. 😍")


                else:
                    uploaded_file = st.file_uploader("[Browse files] 버튼을 클릭 해주세요!", type=["jpg", "png", "jpeg"])

                    st.write("")
                    st.write("")

                    # 저장할 경로 설정
                    SAVE_FOLDER = './data/uploaded_images/'

                    # 폴더가 존재하지 않으면 생성
                    if not os.path.exists(SAVE_FOLDER):
                        os.makedirs(SAVE_FOLDER)

                    # 파일이 업로드된 경우 처리
                    if uploaded_file is not None:
                        # 업로드된 파일을 PIL 이미지로 열기
                        image = Image.open(uploaded_file)

                        # 파일 이름을 가져옴
                        file_name = uploaded_file.name

                        # 저장할 경로 생성
                        file_path = os.path.join(SAVE_FOLDER, file_name)

                        # 이미지 파일을 지정한 경로에 저장
                        image.save(file_path)
                        st.session_state.upload["session"] = 1
                        st.session_state.upload["filepath"] = file_path
                        st.session_state.upload["filename"] = file_name
                        st.text("이미지가 성공적으로 업로드되었습니다.😍")
                        st.write("")

                    ############################ 2. 사용자 두피 이미지 결과  ############################
                    # if uploaded_file is not None and st.session_state.upload["session"] == 1:
                    #     st.markdown("**🔥 사용자 두피 이미지 보기**")
                    #     with st.expander(label="※ 사용자 두피 이미지", expanded=True):
                    #         st.image(image, caption='Uploaded Image.', use_column_width=True)
                    #         # st.write("이미지가 성공적으로 업로드 되었습니다. 😍")

            else:
                uploaded_file = st.file_uploader("[Browse files] 버튼을 클릭 해주세요!", type=["jpg", "png", "jpeg"])

                st.write("")
                st.write("")

                # 저장할 경로 설정
                SAVE_FOLDER = './data/uploaded_images/'

                # 폴더가 존재하지 않으면 생성
                if not os.path.exists(SAVE_FOLDER):
                    os.makedirs(SAVE_FOLDER)

                # 파일이 업로드된 경우 처리
                if uploaded_file is not None:
                    # 업로드된 파일을 PIL 이미지로 열기
                    image = Image.open(uploaded_file)

                    # 파일 이름을 가져옴
                    file_name = uploaded_file.name

                    # 저장할 경로 생성
                    file_path = os.path.join(SAVE_FOLDER, file_name)

                    # 이미지 파일을 지정한 경로에 저장
                    image.save(file_path)
                    st.session_state.upload["session"] = 1
                    st.session_state.upload["filepath"] = file_path
                    st.session_state.upload["filename"] = file_name
                    st.text("이미지가 성공적으로 업로드되었습니다.😍")
                    st.write("")

                ############################ 2. 사용자 두피 이미지 결과  ############################
                # if uploaded_file is not None and st.session_state.upload["session"] == 1:
                #     st.markdown("**🔥 사용자 두피 이미지 보기**")
                #     with st.expander(label="※ 사용자 두피 이미지", expanded=True):
                #         st.image(image, caption='Uploaded Image.', use_column_width=True)
                #         # st.write("이미지가 성공적으로 업로드 되었습니다. 😍")

    # st.button("Home", on_click=home_page, key="button1")
    col3, col4, col5, col6, col7, col8 = st.columns(6)
    with col3:
        st.write("")
    with col4:
        st.write("")
    with col5:
        st.write("")
    with col6:
        if uploaded_file is not None and st.session_state.upload["session"] == 1:
            st.button("Next", on_click=next_page)
    with col7:
        st.write("")
    with col8:
        st.write("1page")

elif st.session_state.page == 1:
    ############################ 3. 사용자 정보 입력하기 ############################
    st.markdown("**🔥 설문조사**")
    st.markdown("* 당신의 두피에 대해 알려주세요. 더 정확한 분석에 도움이 됩니다.")
    st.markdown("* 질문 선택 후 Submit 버튼을 반드시 눌러주세요.")
    # auto_complete = st.toggle("예시 데이터로 채우기")
    with (st.form(key="form")):

        type = st.multiselect(
            label="◾ 질문 1. 당신의 두피 타입을 선택하세요",
            options=list(type_emoji_dict.keys()),
            max_selections=1
            # default=scalp_example["type"] if auto_complete else []
        )

        symptom = st.multiselect(
            label="◾ 질문 2. 당신의 두피 고민/질환 증상이 무엇인가요?",
            options=list(symptom_emoji_dict.keys()),
            max_selections=6
            # default=scalp_example["symptom"] if auto_complete else []
        )

        variety = st.multiselect(
            label="◾ 질문 3. 추천 받기 원하는 제품은 무엇인가요?",
            options=list(variety_emoji_dict.keys()),
            max_selections=7
            # default=scalp_example["variety"] if auto_complete else []
        )

        submit = st.form_submit_button(label="Submit")
        if submit:
            if len(type) == 0:
                st.error("◾ 질문 1.을 선택해주세요.")
            elif len(symptom) == 0:
                st.error("◾ 질문 2.을 선택해주세요.")
            elif len(variety) == 0:
                st.error("◾ 질문 3.을 선택해주세요.")

            else:
                st.success("성공!!!")
                st.session_state.scalp = [{
                    "type": type,
                    "symptom": symptom,
                    "variety": variety,
                    "bidum_state": "",
                    "gakzil_state": "",
                    "hongban_state": "",
                    "nongpo_state": "",
                    "pizy_state": "",
                    "talmo_state": "",
                    "result_type": []
                }]
                st.session_state.survey = 1

                # if "".join(type) == "(건성)두피에 건조함이나 당김을 느낍니다.":
                #     st.markdown("* 당신은 <b>⚜️ [건성 타입]</b>의 두피를 가지셨군요.", unsafe_allow_html=True)
                # elif "".join(type) == "(지성)머리를 감은지 하루 이내에 두피가 기름집니다.":
                #     st.markdown("* 당신은 <b>⚜️ [지성 타입]</b>의 두피를 가지셨군요.", unsafe_allow_html=True)
                #
                # st.markdown(f"* 그리고 당신은 <b>🤦‍♀️️ {','.join(symptom)}</b>를 고민하시는 군요!", unsafe_allow_html=True)
                # st.markdown(f"* 당신을 위해 <b>🧴 {','.join(variety)}</b>를 추천해 드리겠습니다.", unsafe_allow_html=True)

                if "".join(type) == "(건성)두피에 건조함이나 당김을 느낍니다.":
                    st.markdown(f"* 당신의 두피가 **⚜️ 건성 타입**의 **🤦️️ {','.join(symptom)}** 증상이 있다라고 선택하셨습니다.")
                    st.markdown(f"* 다음 페이지에서 당신이 **⚜️ 건성 타입**의 **🤦️️ {','.join(symptom)}** 증상을 가지고 있는 지 **🤠 두피이미지 분석**을 통해 확인해 보도록 하겠습니다.", unsafe_allow_html=True)
                elif "".join(type) == "(지성)머리를 감은지 하루 이내에 두피가 기름집니다.":
                    st.markdown(f"* 당신의 두피가 **⚜️ 지성 타입**의 **🤦️️ {','.join(symptom)}** 증상이 있다라고 선택하셨습니다.")
                    st.markdown(f"* 다음 페이지에서 당신이 **⚜️ 지성 타입**의 **🤦️️ {','.join(symptom)}** 증상을 가지고 있는 지 **🤠 두피이미지 분석**을 통해 확인해 보도록 하겠습니다.", unsafe_allow_html=True)

                st.markdown("* Next 버튼을 눌러주세요!")

    # st.button("Home", on_click=home_page, key="button2")
    col14, col15, col16, col17, col18, col19 = st.columns(6)
    with col14:
        st.write("")
    with col15:
        st.write("")
    with col16:
        st.button("Prev", on_click=prev_main_page)
    with col17:
        if len(type) != 0 and len(symptom) != 0 and len(variety) != 0:
            st.button("Next", on_click=next_page)
        else:
            st.write("")
    with col18:
        st.write("")
    with col19:
        st.write("2page")

elif st.session_state.page == 2:
    # 파일이 업로드된 경우 처리
    if st.session_state.upload["session"] == 1:
        ############################ 4. 예제 두피 이미지 보여주기 ############################
        st.markdown("**🔥 증상 별 두피 이미지**")
        st.image("./data/example.jpg", use_column_width=True)

        # for i in range(1, len(example_scalps_img), 3):
        #     row_scalps = example_scalps_img[i:i+3]
        #     cols = st.columns(3)
        #     for j in range(len(row_scalps)):
        #         with cols[j]:
        #             scalp = row_scalps[j]
        #             with st.expander(label=f"**{i+j}. {scalp['name']}**", expanded=True):
        #                 st.image(scalp["url"], use_column_width=True)

        # cols = st.columns(6)
        # for j in range(1, len(example_scalps_img)):
        #     with cols[j - 1]:
        #         scalp = example_scalps_img[j]
        #         with st.expander(label=f"**{j}. {scalp['name']}**", expanded=True):
        #             st.image(scalp["url"], use_column_width=True)

        cols1, cols2 = st.columns(2)
        with cols1:
            ############################ 5. 사용자 두피 이미지 보여주기 ############################
            st.markdown("**🔥 사용자 두피 이미지 보기**")
            with st.expander(label="클릭시 이미지 확장/삭제", expanded=True):
                st.image(f"./data/uploaded_images/{st.session_state.upload["filename"]}", use_column_width=True)
                st.write("")
                st.write("")
                st.write("")
                st.write("")
                st.markdown(f"* 이미지 파일명 : {st.session_state.upload["filename"]}")
                st.write("")
                st.write("")

        with cols2:
            ############################ 6. 사용자의 두피 상태 결과 보여주기 ############################
            st.markdown("**🔥 사용자의 두피 상태 결과**")

            # 클래스 이름 정의
            class_names = ["👻 양호", "💧 경증", "😈 중증"]  # 클래스

            # 예측 수행 및 결과 출력
            file_path = st.session_state.upload["filepath"]
            pred_class = predict_image(file_path)

            st.session_state.scalp[0]["bidum_state"] = class_names[pred_class[0][0]]
            st.session_state.scalp[0]["gakzil_state"] = class_names[pred_class[1][0]]
            st.session_state.scalp[0]["hongban_state"] = class_names[pred_class[2][0]]
            st.session_state.scalp[0]["nongpo_state"] = class_names[pred_class[3][0]]
            # st.session_state.scalp[0]["pizy_state"] = class_names[pred_class[4][0]]
            st.session_state.scalp[0]["talmo_state"] = class_names[pred_class[5][0]]
            if class_names[pred_class[1][0]] == "👻 양호":
                st.session_state.scalp[0]["pizy_state"] = "😈 중증"
            elif class_names[pred_class[1][0]] == "💧 경증":
                st.session_state.scalp[0]["pizy_state"] = "💧 경증"
            else:
                st.session_state.scalp[0]["pizy_state"] = "👻 양호"

            with st.expander(label="클릭시 이미지 확장/삭제", expanded=True):
                st.markdown(f"<p style='font-size:15px;'><b>1. 비듬 : {class_names[pred_class[0][0]]}<b></p>", unsafe_allow_html=True)
                st.markdown(f"<p style='font-size:12px;'>[양호({round(pred_class[0][1]*100)}%), 경증({round(pred_class[0][2]*100)}%), 중증({round(pred_class[0][3]*100)}%)]</p>", unsafe_allow_html=True)
                st.markdown(f"<p style='font-size:15px;'><b>2. 모낭사이홍반 : {class_names[pred_class[2][0]]}<b></p>", unsafe_allow_html=True)
                st.markdown(f"<p style='font-size:12px;'>[양호({round(pred_class[2][1]*100)}%), 경증({round(pred_class[2][2]*100)}%), 중증({round(pred_class[2][3]*100)}%)]</p>", unsafe_allow_html=True)
                st.markdown(f"<p style='font-size:15px;'><b>3. 모낭홍반농포 : {class_names[pred_class[3][0]]}<b></p>", unsafe_allow_html=True)
                st.markdown(f"<p style='font-size:12px;'>[양호({round(pred_class[3][1]*100)}%), 경증({round(pred_class[3][2]*100)}%), 중증({round(pred_class[3][3]*100)}%)]</p>", unsafe_allow_html=True)
                st.markdown(f"<p style='font-size:15px;'><b>4. 탈모 : {class_names[pred_class[5][0]]}<b></p>", unsafe_allow_html=True)
                st.markdown(f"<p style='font-size:12px;'>[양호({round(pred_class[5][1]*100)}%), 경증({round(pred_class[5][2]*100)}%), 중증({round(pred_class[5][3]*100)}%)]</p>", unsafe_allow_html=True)
                st.markdown(f"<p style='font-size:15px;'><b>5. 미세각질 : {class_names[pred_class[1][0]]}<b></p>", unsafe_allow_html=True)
                st.markdown(f"<p style='font-size:12px;'>[양호({round(pred_class[1][1]*100)}%), 경증({round(pred_class[1][2]*100)}%), 중증({round(pred_class[1][3]*100)}%)]</p>", unsafe_allow_html=True)
                st.markdown(f"<p style='font-size:15px;'><b>6. 피지과다 : {st.session_state.scalp[0]["pizy_state"]}<b></p>", unsafe_allow_html=True)
                # st.markdown(f"[**양호**({round(pred_class[4][1]*100)}%), **경증**({round(pred_class[4][2]*100)}%), **중증**({round(pred_class[4][3]*100)}%)]")

        with st.expander(label="상태분석 노트", expanded=True):

            # 입력값 내부 변수에 저장
            bidum_state = st.session_state.scalp[0]["bidum_state"]
            gakzil_state = st.session_state.scalp[0]["gakzil_state"]
            hongban_state = st.session_state.scalp[0]["hongban_state"]
            nongpo_state = st.session_state.scalp[0]["nongpo_state"]
            pizy_state = st.session_state.scalp[0]["pizy_state"]
            talmo_state = st.session_state.scalp[0]["talmo_state"]

            class_names = ["👻 양호", "💧 경증", "😈 중증"]  # 클래스

            tmp = []
            tmp2 = []
            if bidum_state == class_names[2] or gakzil_state == class_names[2] or hongban_state == class_names[2] or nongpo_state == class_names[2] or pizy_state == class_names[2] or talmo_state == class_names[2]:
                if bidum_state == class_names[2]:
                    tmp.append("비듬")
                    tmp2.append(f"비듬 : {bidum_state}")
                if gakzil_state == class_names[2]:
                    tmp.append("미세각질")
                    tmp2.append(f"미세각질 : {gakzil_state}")
                if hongban_state == class_names[2]:
                    tmp.append("모낭사이홍반")
                    tmp2.append(f"모낭사이홍반 : {hongban_state}")
                if nongpo_state == class_names[2]:
                    tmp.append("모낭홍반농포")
                    tmp2.append(f"모낭홍반농포 : {nongpo_state}")
                if pizy_state == class_names[2]:
                    tmp.append("피지과다")
                    tmp2.append(f"피지과다 : {pizy_state}")
                if talmo_state == class_names[2]:
                    tmp.append("탈모")
                    tmp2.append(f"탈모 : {talmo_state}")
            elif bidum_state == class_names[1] or gakzil_state == class_names[1] or hongban_state == class_names[1] or nongpo_state == class_names[1] or pizy_state == class_names[1] or talmo_state == class_names[1]:
                if bidum_state == class_names[1]:
                    tmp.append("비듬")
                    tmp2.append(f"비듬 : {bidum_state}")
                if gakzil_state == class_names[1]:
                    tmp.append("미세각질")
                    tmp2.append(f"미세각질 : {gakzil_state}")
                if hongban_state == class_names[1]:
                    tmp.append("모낭사이홍반")
                    tmp2.append(f"모낭사이홍반 : {hongban_state}")
                if nongpo_state == class_names[1]:
                    tmp.append("모낭홍반농포")
                    tmp2.append(f"모낭홍반농포 : {nongpo_state}")
                if pizy_state == class_names[1]:
                    tmp.append("피지과다")
                    tmp2.append(f"피지과다 : {pizy_state}")
                if talmo_state == class_names[1]:
                    tmp.append("탈모")
                    tmp2.append(f"탈모 : {talmo_state}")

            # st.markdown(f"* 당신의 두피 상태는 **{",".join(tmp2)}** 입니다.")

            df = load_data(variety="shampoo")
            result = product_recommend(df)
            result_type = st.session_state.scalp[0]["result_type"]
            # st.markdown(f"* 그 결과 당신은 <b>⚜️{",".join(result_type)}</b> 증상으로 분석되었어요.", unsafe_allow_html=True)

            survey_type = st.session_state.scalp[0]["type"]
            survey_symptom = st.session_state.scalp[0]["symptom"]
            survey_variety = st.session_state.scalp[0]["variety"]


            if "".join(survey_type) == "(지성)머리를 감은지 하루 이내에 두피가 기름집니다.":
                for t in result_type:
                    if t == "지성":
                        st.markdown(f"* 설문조사에서 당신은 **⚜️ 지성 타입**이라고 선택하셨습니다. 두피이미지 분석 결과 **⚜️ {",".join(result_type)}타입**으로 판단됩니다.")
                    elif t == "건성":
                        st.markdown(f"* 설문조사에서 당신은 **⚜️ 지성 타입**이라고 선택하셨습니다. 그러나 두피이미지 분석 결과 **⚜️ {",".join(result_type)}타입*으로 판단됩니다.")
            elif "".join(survey_type) == "(건성)두피에 건조함이나 당김을 느낍니다.":
                for t in result_type:
                    if t == "건성":
                        st.markdown(f"* 설문조사에서 당신은 **⚜️ 건성 타입**이라고 선택하셨습니다. 두피이미지 분석 결과 **⚜️ {",".join(result_type)}타입**으로 판단됩니다.")
                    elif t == "지성":
                        st.markdown(f"* 설문조사에서 당신은 **⚜️ 건성 타입**이라고 선택하셨습니다. 그러나 두피이미지 분석 결과 **⚜️ {",".join(result_type)}타입**으로 판단됩니다.")
            else:
                st.markdown(str(survey_type))

            # for s in survey_symptom:
            #     if s == "비듬":
            #         for i, t in enumerate(tmp):
            #             if t == "비듬":
            #                 st.markdown(f"* 고민하신 것 처럼 두피 이미지 분석 결과 **비듬**이 나왔고 상태는 **{bidum_state}**입니다.")
            #                 break
            #             # else:
            #             #     if i == len(tmp)-1:
            #             #         st.markdown(f"* **비듬**을 고민하셨죠. 그만 고민하세요. 두피 이미지 결과에 따라 **비듬**은 아니세요.")
            #     if s == "미세각질":
            #         for i, t in enumerate(tmp):
            #             if t == "미세각질":
            #                 st.markdown(f"* 고민하신 것 처럼 두피 이미지 분석 결과 **미세각질**이 나왔고 상태는 **{gakzil_state}**입니다.")
            #                 break
            #             # else:
            #             #     if i == len(tmp) - 1:
            #             #         st.markdown(f"* **미세각질**을 고민하셨죠. 그만 고민하세요. 두피 이미지 결과에 따라 **미세각질**은 아니세요.")
            #     if s == "모낭사이홍반":
            #         for i, t in enumerate(tmp):
            #             if t == "모낭사이홍반":
            #                 st.markdown(f"* 고민하신 것 처럼 두피 이미지 분석 결과 **모낭사이홍반**이 나왔고 상태는 **{hongban_state}**입니다.")
            #                 break
            #             # else:
            #             #     if i == len(tmp)-1:
            #             #         st.markdown(f"* **모낭사이홍반**를 고민하셨죠. 그만 고민하세요. 두피 이미지 결과에 따라 **모낭사이홍반**는 아니세요.")
            #     if s == "모낭홍반농포":
            #         for i, t in enumerate(tmp):
            #             if t == "모낭홍반농포":
            #                 st.markdown(f"* 고민하신 것 처럼 두피 이미지 분석 결과 **모낭홍반농포**가 나왔고 상태는 **{nongpo_state}**입니다.")
            #                 break
            #             # else:
            #             #     if i == len(tmp) - 1:
            #             #         st.markdown(f"* **모낭홍반농포**를 고민하셨죠. 그만 고민하세요. 두피 이미지 결과에 따라 **모낭홍반농포**는 아니세요.")
            #     if s == "피지과다":
            #         for i, t in enumerate(tmp):
            #             if t == "피지과다":
            #                 st.markdown(f"* 고민하신 것 처럼 두피 이미지 분석 결과 **피지과다**가 나왔고 상태는 **{pizy_state}**입니다.")
            #                 break
            #             # else:
            #             #     if i == len(tmp) - 1:
            #             #         st.markdown(f"* **피지과다**를 고민하셨죠. 그만 고민하세요. 두피 이미지 결과에 따라 **피지과다**는 아니세요.")
            #     if s == "탈모":
            #         for i, t in enumerate(tmp):
            #             if t == "탈모":
            #                 st.markdown(f"* 고민하신 것 처럼 두피 이미지 분석 결과 **탈모**가 나왔고 상태는 **{talmo_state}**입니다.")
            #                 break
            #             # else:
            #             #     if i == len(tmp)-1:
            #             #         st.markdown(f"* **탈모**를 고민하셨죠. 그만 고민하세요. 두피 이미지 결과에 따라 **탈모**는 아니세요.")

            st.markdown(f"* 다른 증상에 대해서는 고민하지 마세요. 두피 이미지 분석 결과에 따라 당신은 **{",".join(tmp2)}**으로 판단됩니다.")
            st.markdown(f"* 다음 페이지에서 **🤠 {",".join(tmp)}**의 **원인과 특징과 관리방안**에 대해서 알아보겠습니다.")
            st.markdown(f"* 그 다음 페이지에서 **⚜️ {",".join(result_type)}** 타입에 맞는 **🧴 {"".join(survey_variety)}**들을 추천해 드리겠습니다.")
            st.markdown("* Next 버튼을 눌러주세요!")

        col5, col6, col7, col8, col9, col10 = st.columns(6)
        with col5:
            st.button("Home", on_click=home_page, key="button4")
        with col6:
            st.write("")
        with col7:
            st.button("Prev", on_click=prev_page)
        with col8:
            st.button("Next", on_click=next_page)
        with col9:
            st.write("")
        with col10:
            st.write("3page")

elif st.session_state.page == 3:
    ############################ 7. 두피 타입별 원인과 특징 그리고 관리방안을 보여주기 ############################
    st.write("")
    st.markdown("**🔥 두피 타입별 원인과 특징 그리고 관리방안**")

    # 입력값 내부 변수에 저장
    bidum_state = st.session_state.scalp[0]["bidum_state"]
    gakzil_state = st.session_state.scalp[0]["gakzil_state"]
    hongban_state = st.session_state.scalp[0]["hongban_state"]
    nongpo_state = st.session_state.scalp[0]["nongpo_state"]
    pizy_state = st.session_state.scalp[0]["pizy_state"]
    talmo_state = st.session_state.scalp[0]["talmo_state"]


    class_names = ["👻 양호", "💧 경증", "😈 중증"]  # 클래스

    scalp_type = []
    tmp = []
    if bidum_state == class_names[2] or gakzil_state == class_names[2] or hongban_state == class_names[2] or nongpo_state == class_names[2] or pizy_state == class_names[2] or talmo_state == class_names[2]:
        if bidum_state == class_names[2]:
            tmp.append("비듬")
            scalp_type.append("비듬")
        if gakzil_state == class_names[2]:
            tmp.append("미세각질")
            scalp_type.append("미세각질")
        if hongban_state == class_names[2]:
            tmp.append("모낭사이홍반")
            scalp_type.append("모낭사이홍반")
        if nongpo_state == class_names[2]:
            tmp.append("모낭홍반농포")
            scalp_type.append("모낭홍반농포")
        if pizy_state == class_names[2]:
            tmp.append("피지과다")
            scalp_type.append("피지과다")
        if talmo_state == class_names[2]:
            tmp.append("탈모")
            scalp_type.append("탈모")
    elif bidum_state == class_names[1] or gakzil_state == class_names[1] or hongban_state == class_names[1] or nongpo_state == class_names[1] or pizy_state == class_names[1] or talmo_state == class_names[1]:
        if bidum_state == class_names[1]:
            tmp.append("비듬")
            scalp_type.append("비듬")
        if gakzil_state == class_names[1]:
            tmp.append("미세각질")
            scalp_type.append("미세각질")
        if hongban_state == class_names[1]:
            tmp.append("모낭사이홍반")
            scalp_type.append("모낭사이홍반")
        if nongpo_state == class_names[1]:
            tmp.append("모낭홍반농포")
            scalp_type.append("모낭홍반농포")
        if pizy_state == class_names[1]:
            tmp.append("피지과다")
            scalp_type.append("피지과다")
        if talmo_state == class_names[1]:
            tmp.append("탈모")
            scalp_type.append("탈모")

    st.markdown(f"* 당신의 두피 상태인 **🤠 {",".join(tmp)}**의 원인과 특징 그리고 관리방안을 확인하도록 하겠습니다.")

    with st.spinner('두피 타입의 원인과 특징 그리고 관리방안을 보여 주고 있습니다...'):
        prompt = generate_prompt(','.join(scalp_type))
        response = request_chat_completion(prompt)
    print_streaming_response(response)

    st.markdown("* Next 버튼을 눌러주세요!")

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.button("Home", on_click=home_page, key="button5")
    with col2:
        st.write("")
    with col3:
        st.button("Prev", on_click=prev_page)
    with col4:
        st.button("Next", on_click=next_page)
    with col5:
        st.write("")
    with col6:
        st.write("4page")

elif st.session_state.page == 4:
    ############################ 8. 추천 제품 목록 보여주기  ############################

    # 입력값 내부 변수에 저장
    bidum_state = st.session_state.scalp[0]["bidum_state"]
    gakzil_state = st.session_state.scalp[0]["gakzil_state"]
    hongban_state = st.session_state.scalp[0]["hongban_state"]
    nongpo_state = st.session_state.scalp[0]["nongpo_state"]
    pizy_state = st.session_state.scalp[0]["pizy_state"]
    talmo_state = st.session_state.scalp[0]["talmo_state"]

    variety = st.session_state.scalp[0]["variety"]

    st.write("")
    st.markdown("**🔥 추천 제품 목록**")

    class_names = ["👻 양호", "💧 경증", "😈 중증"]  # 클래스

    tmp = []
    if bidum_state == class_names[2] or gakzil_state == class_names[2] or hongban_state == class_names[2] or nongpo_state == class_names[2] or pizy_state == class_names[2] or talmo_state == class_names[2]:
        if bidum_state == class_names[2]:
            tmp.append(f"비듬 : {bidum_state}")
        if gakzil_state == class_names[2]:
            tmp.append(f"미세각질 : {gakzil_state}")
        if hongban_state == class_names[2]:
            tmp.append(f"모낭사이홍반 : {hongban_state}")
        if nongpo_state == class_names[2]:
            tmp.append(f"모낭홍반농포 : {nongpo_state}")
        if pizy_state == class_names[2]:
            tmp.append(f"피지과다 : {pizy_state}")
        if talmo_state == class_names[2]:
            tmp.append(f"탈모 : {talmo_state}")
    elif bidum_state == class_names[1] or gakzil_state == class_names[1] or hongban_state == class_names[1] or nongpo_state == class_names[1] or pizy_state == class_names[1] or talmo_state == class_names[1]:
        if bidum_state == class_names[1]:
            tmp.append(f"비듬 : {bidum_state}")
        if gakzil_state == class_names[1]:
            tmp.append(f"미세각질 : {gakzil_state}")
        if hongban_state == class_names[1]:
            tmp.append(f"모낭사이홍반 : {hongban_state}")
        if nongpo_state == class_names[1]:
            tmp.append(f"모낭홍반농포 : {nongpo_state}")
        if pizy_state == class_names[1]:
            tmp.append(f"피지과다 : {pizy_state}")
        if talmo_state == class_names[1]:
            tmp.append(f"탈모 : {talmo_state}")

    st.markdown(f"* 당신의 두피 상태는 **{",".join(tmp)}** 이고,")

    df = load_data(variety="shampoo")
    result = product_recommend(df)
    result_type = st.session_state.scalp[0]["result_type"]
    st.markdown(f"* 그 결과 당신은 <b>⚜️{",".join(result_type)}</b> 타입으로 분석되었습니다.", unsafe_allow_html=True)

    for v in variety:

        if v == "샴푸":
            st.write("")
            # st.text("* 샴푸를 추천해드리겠습니다.")
            df_shampoo = load_data(variety="shampoo")
            result_shampoo = product_recommend(df_shampoo)
            find_shampoo = product_find(result_shampoo, "샴푸")
            product_view(find_shampoo)
            product_description(find_shampoo)
        if v == "린스/컨디셔너":
            st.write("")
            # st.text("* 린스/컨디셔너를 추천해드리겠습니다.")
            df_rinse = load_data(variety = "rinse")
            result_rinse = product_recommend(df_rinse)
            find_rinse = product_find(result_rinse, "린스/컨디셔너")
            product_view(find_rinse)
            product_description(find_rinse)
        if v == "샴푸바/드라이샴푸":
            st.write("")
            # st.text("* 샴푸바/드라이샴푸를 추천해드리겠습니다.")
            df_bar = load_data(variety="bar")
            result_bar = product_recommend(df_bar)
            find_bar = product_find(result_bar, "샴푸바/드라이샴푸")
            product_view(find_bar)
            product_description(find_bar)
        if v == "헤어오일/헤어세럼":
            st.write("")
            # st.text("* 헤어오일/헤어세럼을 추천해드리겠습니다.")
            df_hairoil = load_data(variety="hairoil")
            result_hairoil = product_recommend(df_hairoil)
            find_hairoil = product_find(result_hairoil, "헤어오일/헤어세럼")
            product_view(find_hairoil)
            product_description(find_hairoil)
        if v == "헤어워터":
            st.write("")
            # st.text("* 헤어워터를 추천해드리겠습니다.")
            df_hairwater = load_data(variety="hairwater")
            result_hairwater = product_recommend(df_hairwater)
            find_hairwater = product_find(result_hairwater, "헤어워터")
            product_view(find_hairwater)
            product_description(find_hairwater)
        if v == "두피팩/스케일러":
            st.write("")
            # st.text("* 두피팩/스케일러를 추천해드리겠습니다.")
            df_scaler = load_data(variety="scaler")
            result_scaler = product_recommend(df_scaler)
            find_scaler = product_find(result_scaler, "두피팩/스케일러")
            product_view(find_scaler)
            product_description(find_scaler)
        if v == "헤어토닉/두피토닉":
            st.write("")
            # st.text("* 헤어토닉/두피토닉을 추천해드리겠습니다.")
            df_tonic = load_data(variety="tonic")
            result_tonic = product_recommend(df_tonic)
            find_tonic = product_find(result_tonic, "헤어토닉/두피토닉")
            product_view(find_tonic)
            product_description(find_tonic)

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.button("Home", on_click=home_page, key="button6")
    with col2:
        st.write("")
    with col3:
        st.button("Prev", on_click=prev_page)
    with col4:
        st.write("")
    with col5:
        st.write("")
    with col6:
        st.write("5page")