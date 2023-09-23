import requests
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering

processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-capfilt-large")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-capfilt-large")


def generate_text_prompt(raw_image):
    """
    Parameters
    ----------
    raw_image: np.array
               Input prompt image 

    Returns
    -------
    text_prompt: str
               Optimized prompt string

    We generate a new prompt string for the Stable Diffusion model
    For the new prompt string, three key info are extracted from the 
    prompt image:
    1. Colors in the original image
    2. Style of the original image
    3. Whether the design was minimalist
    """
    colors = "What are the colors present in this image?"
    inputs = processor(raw_image, colors, return_tensors="pt")

    out = model.generate(**inputs)
    colors_out = processor.decode(out[0], skip_special_tokens=True)
   
    style = "Explain the style of this image"
    inputs = processor(raw_image, style, return_tensors="pt")

    out = model.generate(**inputs)
    style_out = processor.decode(out[0], skip_special_tokens=True)


    minimalist = "Is it a minimalistic design?"
    inputs = processor(raw_image, minimalist, return_tensors="pt")

    out = model.generate(**inputs)
    minimalist_out = processor.decode(out[0], skip_special_tokens=True)

    if minimalist_out:
        minimalist_out = 'minimalist'
    else:
        minimalist_out = ''

    text_prompt = f"Generate a {minimalist_out} UI design template that has {colors_out} colors with a {style_out} style" 
    return text_prompt