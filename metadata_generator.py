import pytesseract
from PIL import Image
import sys
import json
import numpy as np 

def metadata_generation(image_array):
    img = Image.fromarray(image_array)
    width, height = img.size
    design_metadata={
        'frame_width': width,
        'frame_height': height,
        'text_areas' : []
    }

    text_data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    for i, word_info in enumerate(text_data['text']):
        word = word_info.strip()
        confidence = int(text_data['conf'][i])
        x, y, w, h = text_data['left'][i], text_data['top'][i], text_data['width'][i], text_data['height'][i]
        #print(f"Word: {word}, Confidence: {confidence}, Coordinates: (x:{x}, y:{y}, w:{w}, h:{h})")
        if(word!= '' and word!=' '):
            word_json={
                'Word': word,
                'Confidence' : confidence,
                'Coordinates' : f'(x:{x}, y:{y}, w:{w}, h:{h})'
            }
            design_metadata['text_areas'].append(word_json)

    # with open('design_metadata.json', "w") as json_file:
    #     json.dump(design_metadata, json_file)

    return design_metadata
    

# try:
#     if(sys.argv[0] != ''):
#         metadata_generation(sys.argv[1]) 
#     else:
#         print('Error: No Input Image given')
# except:
#     print('Oops! Server Error')
