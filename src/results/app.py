
import streamlit as st
from apps import home, crop_disease_detector, crop_detector, weather_predictor
from PIL import Image

with st.container():
    proj_title_col, logo_col = st.columns([6,1])

    with proj_title_col:
      st.header('Improving Food Security and Crop Yield in Nigeria through Machine Learning')

    with logo_col:
      logo = Image.open('../osun_chapter.png')
      logo = logo.resize((75,75))
      st.image(logo)

PAGES = {
    "Home": home,
    "Crop Disease Detection": crop_disease_detector,
    'Crop Classification': crop_detector,
    'Weather Forecast': weather_predictor
}

selection = st.sidebar.radio("Menu", list(PAGES.keys()))
page = PAGES[selection]
page.app()