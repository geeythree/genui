import streamlit as st
from image_generator import create_design
from metadata_generator import metadata_generation
import json

uploaded_file = st.file_uploader("Choose a image file")

if uploaded_file is not None:
    progress_text = "Generating template. Please wait..."
    with st.spinner(progress_text):
        file_bytes = uploaded_file.read()
        image = create_design(file_bytes)
    st.subheader('Suggested Design Template: ')
    #btn = st.button('Generate Meta-Data for the image',metadata_generation(image))
    try:
        json_file = str(metadata_generation(image))
    except:
        json_file = str({})
    # json.dumps(json_file, ensure_ascii=False).encode('utf8')
    st.download_button(
    label="Generate Meta-Data for the image",
    data=json_file,
    file_name='ui_metadata.json',
    #mime='text/plain',
    )
    st.image(image, channels="BGR")
    #if btn: 
    #    json_file = metadata_generation(image)

