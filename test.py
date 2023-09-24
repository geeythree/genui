import requests
from PIL import Image
from transformers import BlipProcessor, Blip2ForConditionalGeneration, BlipForQuestionAnswering

processor = BlipProcessor.from_pretrained("Salesforce/blip2-flan-t5-xxl")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xxl")

# img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg' 
# raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

# processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-capfilt-large")
# model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-capfilt-large")


image_path = 'moodboards/3-dating-app-mood-board.png'
raw_image = Image.open(image_path).convert('RGB')

question = "List all the extractible contents on this image"
inputs = processor(raw_image, question, return_tensors="pt")

out = model.generate(**inputs)
print(processor.decode(out[0], skip_special_tokens=True))
