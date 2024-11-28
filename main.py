import onnxruntime as ort
import streamlit as st
from PIL import Image
import numpy as np
import os
from tensorflow.keras.preprocessing.image import img_to_array


# โหลดโมเดล ONNX
onnx_model_path = "./model.onnx"
session = ort.InferenceSession(onnx_model_path)

#-----------------------------------------------------------------------------------------------------------------
# cut_image.py
def cut_image(image):
    width, height = image.size
    frac=0.6
    crop_left_width = int(width * frac)
    cropped_left = image.crop((0, 0, crop_left_width, height))
    crop_right_width = width - crop_left_width
    cropped_right = image.crop((crop_right_width, 0, width, height)).transpose(Image.FLIP_LEFT_RIGHT)
    return cropped_left, cropped_right
#---------------------------------------------------------------------------------------------------------------------

# ฟังก์ชันการทำนาย
def predict_image(img, session, height, width):
    img = img.resize((height, width))
    x = img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)

    input_name = session.get_inputs()[0].name
    output_name_regression = session.get_outputs()[0].name
    output_name_classification = session.get_outputs()[1].name

    regression_result = session.run([output_name_regression], {input_name: x})[0]
    classification_result = session.run([output_name_classification], {input_name: x})[0]

    return regression_result, classification_result
#-----------------------------------------------------------------------------------------------------------------------------------
#หาค่า confident
def calculate_confident(value):
    if value >= 0.5: #male
        confident = value
    else:
        confident = 1 - value #female
    return confident

#-----------------------------------------------------------------------------------------------------------------
# ฟังก์ชันหาค่าอายุและเพศ
def predict_age_gender(img_paths):
    pred_list_regression = []
    pred_list_classification = []

    for img in img_paths:
        height = width = session.get_inputs()[0].shape[2]
        regression_result, classification_result = predict_image(img, session, height, width)
        pred_list_regression.append(regression_result)
        pred_list_classification.append(classification_result)

    con_0 = calculate_confident(pred_list_classification[0][0])
    con_1 = calculate_confident(pred_list_classification[1][0])

    if con_0 > con_1:
        gender_pred = "Male" if pred_list_classification[0][0] >= 0.5 else "Female"
        age_pred = np.around(pred_list_regression[0][0])
    else:
        gender_pred = "Male" if pred_list_classification[1][0] >= 0.5 else "Female"
        age_pred = np.around(pred_list_regression[1][0])

    confidence = max(con_0, con_1)
    return age_pred, gender_pred, confidence

#---------------------------------------------------------------------------------
# เก็บรูป
import os
if not os.path.exists("uploads"):
    os.makedirs("uploads")
#-------------------------------------------------------------------------------------
# เพิ่ม Open Graph Metadata
st.markdown("""
<head>
    <meta property="og:title" content="DeeptoothDuo: AI for Dental X-rays" />
    <meta property="og:description" content="Estimate age and sex from panoramic dental X-rays." />
    <meta property="og:image" content=".Doc/logo1.svg" />
    <meta property="og:url" content="https://deeptoothduo.streamlit.app/" />
</head>
""", unsafe_allow_html=True)


# UI
st.markdown("<h1 style='text-align: center;'> 🦷 Age and Sex Estimation via Panoramic X-ray Image</h1>", unsafe_allow_html=True)
st.write("") 
st.write("") 

# รับภาพจากผู้ใช้
uploaded_file = st.file_uploader("Choose a dental X-ray image", type=["jpg", "jpeg", "png"])


if uploaded_file is not None:
    image = Image.open(uploaded_file)
    filename = uploaded_file.name  # Get the filename
    st.image(image, caption=f"Uploaded X-ray image: {filename}", use_column_width=True)

   # บันทึกภาพลงในโฟลเดอร์ "uploads"
    image_path = os.path.join("uploads", filename)
    image.save(image_path)

    # ตัดภาพเป็นภาพซ้ายและขวา
    left_img, right_img = cut_image(image)
    
    # เรียกฟังก์ชันทำนายอายุและเพศจากภาพทั้งสอง
    age, gender, confidence = predict_age_gender([left_img, right_img])

    # แสดงผลการทำนาย
    st.subheader("Prediction Results")
    st.write(f"<span style='font-size:24px;'> <b>Age</b>: <span style='color:green;'><b>{int(age)}</b></span><span style='color:blue;'>± 1.96</span></span><span style='font-size:20px;'> years</span>", unsafe_allow_html=True)
    st.write(f"<span style='font-size:24px;'> <b>Sex</b>: <span style='color:green;'><b>{gender}</b></span> </span><span style='font-size:20px;'>(Confidence: <span style='color:blue;'>{confidence.item()*100:.2f}%</span>) </span>", unsafe_allow_html=True)


    # เพิ่มส่วนแสดง BibTeX Citation
    st.markdown("<h2 style='text-align: center;'>Visit our publication</h2>", unsafe_allow_html=True)
    st.markdown("""
    <div style="background-color: #252530; border: 1px solid #ddd; padding: 15px; font-family: monospace; font-size: 14px; line-height: 1.6;">
    Hirunchavarod, N., Phuphatham, P., Sributsayakarn, N., Prathansap, N., Pornprasertsuk-Damrongsri, S., Jirarattanasopha, V. and Intharah, T., 2024, May. Deeptoothduo: Multi-task age-sex estimation and understanding via panoramic radiograph. In 2024 IEEE International Symposium on Biomedical Imaging (ISBI) (pp. 1-5). IEEE.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<h2 style='text-align: center;'>Visit our project page</h2>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center;">
        <a href="https://nattntn.github.io/OPG_SHAPer_webpage/" target="_blank" 
            style="font-size: 20px; color: #007bff; text-decoration: none;">
            🌐 Click here
        </a>
    </div>
""", unsafe_allow_html=True)