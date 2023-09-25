import requests
import base64
import io
from blip_vqa import generate_text_prompt
import numpy as np
from cv2 import imdecode, imencode, COLOR_RGB2BGR, cvtColor, INTER_AREA, resize, imshow
import os
import cv2
from dotenv import load_dotenv
from imageio import imread
import imutils

load_dotenv()

def resize_image(raw_image):
    # if raw_image.shape[0] != 1024 and raw_image.shape[1] != 1024:
    #     resized= resize(raw_image,(1024,1024))
    #     cv2.imwrite('test.png', resized)
        
    #     offset_y, offset_x = ((1024 - raw_image.shape[0]) // 2, (1024 - raw_image.shape[1]) // 2)
    #     return offset_y,offset_x,resized
    # else:
    #     return 1024,1024,raw_image
        
    if raw_image.shape[0] < 1024 and raw_image.shape[1] < 1024:
        im = np.zeros((1024,1024,3), np.uint8)
        offset_y, offset_x = ((1024 - raw_image.shape[0]) // 2, (1024 - raw_image.shape[1]) // 2)
        im[offset_y:offset_y+raw_image.shape[0], offset_x:offset_x+raw_image.shape[1]] = raw_image
        # img_encode = imencode('.png', im)[1]
        
        return offset_y, offset_x, im
    
    elif raw_image.shape[1] > 1024 or  raw_image.shape[0] > 1024:
        if raw_image.shape[1] > 1024:
            resized = imutils.resize(raw_image, width=1023)
            if resized.shape[0] > 1023:
                resized = imutils.resize(resized, height=1023)
        else:
            resized = imutils.resize(raw_image, height=1023)
            if resized.shape[1] > 1023:
                resized = imutils.resize(resized, width=1023)
        offset_y, offset_x = ((1024 - resized.shape[0]) // 2, (1024 - resized.shape[1]) // 2)
        return offset_y, offset_x, resized
    else:
        return 1024, 1024, raw_image

def encode_image(image):
    return imencode('.png',image)[1]

def create_design(raw_image):
    """
    Parameters
    ----------
    raw_image: str
               base64 encoded string of the image

    Returns
    -------
    image: np.array
          Final design suggestion image
    """
    file_bytes = np.asarray(bytearray(raw_image), dtype=np.uint8)
    raw_image = imdecode(file_bytes, 1)

    # BLIP model required the input dimensions of image to be 1024x1024 
    # Insted od resizing the image, paste it on a new image of dim 1024x1024
    offset_y, offset_x, resized_img = resize_image(raw_image)
    # print(f"Resized 1: {resized_img.shape}")
    if resized_img.shape[0] < 1024 or resized_img.shape[1] < 1024:
        offset_y, offset_x, resized_img = resize_image(resized_img)
    img_encode = encode_image(resized_img)

    # Model pipeline 1: generate text prompt
    text_prompt = generate_text_prompt(raw_image)

    # Model pipeline 2:  generate ui suggestions with text + image prompt
    response = requests.post(
        "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/image-to-image",
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {os.getenv('STABILITY_KEY')}"
        },
        files={
            "init_image": img_encode 
        },
        data={
            "init_image_mode": "IMAGE_STRENGTH",
            "image_strength": 0.15,
            "steps": 40,
            "seed": 0,
            "cfg_scale": 5,
            "samples": 1,
            "text_prompts[0][text]": text_prompt,
            "text_prompts[0][weight]": 1,
            "text_prompts[1][text]": 'blurry, bad',
            "text_prompts[1][weight]": -1,
        }
    )

    if response.status_code != 200:
        raise Exception("Non-200 response: " + str(response.text))

    data = response.json()

    # decode the returned image strings
    images = [base64.b64decode(image["base64"]) for image in data["artifacts"]]

    # crop out the blank background
    image = imread(io.BytesIO(images[0])) # images[0] since we are generating only one suggestion
    image = cvtColor(image, COLOR_RGB2BGR)
    image = image[offset_y:offset_y+raw_image.shape[0], offset_x:offset_x+raw_image.shape[1]]
    
    return  image

