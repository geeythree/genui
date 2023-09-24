import streamlit as st
from image_generator import create_design
from metadata_generator import metadata_generation

# OPENAI KEY - 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

uploaded_file = st.file_uploader("Choose a image file")

if uploaded_file is not None:
    progress_text = "Generating template. Please wait..."
    with st.spinner(progress_text):
        file_bytes = uploaded_file.read()
        image = create_design(file_bytes)
    st.subheader('Suggested Design Template: ')
    st.button('Generate Meta-Data for the image',metadata_generation(image))
    st.image(image, channels="BGR")
