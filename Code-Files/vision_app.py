import os

from PIL import Image
from dotenv import load_dotenv
import streamlit as st
import base64
import google.generativeai as genai

load_dotenv()  # loading the environment variables

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# func to load gemini pro vision model and get responses

model = genai.GenerativeModel("gemini-pro-vision")

def get_response(input, image):
    if input!="":
        response = model.generate_content([input, image])
    else:
        response = model.generate_content(image)

    return response.text


## initialze streamlit app

st.set_page_config(page_title=" Question Answer Demo")

# Function to set a background image
def set_background(image_file):
    with open(image_file, "rb") as image:
        b64_image = base64.b64encode(image.read()).decode("utf-8")
    css = f"""
    <style>
    .stApp {{
        background: url(data:image/png;base64,{b64_image});
        background-size: cover;
        background-position: centre;
        backgroun-repeat: no-repeat;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Set the background image
set_background("assets/background_image.png")

st.header("Gemini-Decode : VLM")
input = st.text_input("Input : ", key = "input")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
image = ""

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)

submit = st.button("Submit")

# when submit button is clicked

if submit:
    response = get_response(input, image)

    st.subheader("Bot Response : ")
    st.write(response)