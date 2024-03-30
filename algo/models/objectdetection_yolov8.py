#pip3 install ultralytics

from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import cv2
import time
from PIL import Image
import numpy as np
from pathlib import Path
import os
from models.obj_class import SYMBOL_MAP

# Load the YOLOv8 model
model = YOLO("/home/pi/MDP2/src/Algo-main/yolov8_ft.pt")
classes = model.names

def detect(path):
  
  # Check if image loaded properly
  print("detect")
  img = cv2.imread(path)
  
  if img is None:
    print("Image is incorrect. Check image path")
    return
  
  # Preprocessing
  image_path = preprocess(path, 640, 640)
  # Perform object detection
  results = model(image_path)[0]  
  # Visualize results 
  #cv2.imshow("YOLOv8 Detection", results.plot())  
  filename = path[path.rindex('/')+1:]
  #results.save(f"/results/{filename}")
  
  img = cv2.imread(image_path)
  # Draw box
  boxes = results.boxes.xywh
  clss = results.boxes.cls
  annotator = Annotator(img, line_width=2, example=str(model.names))
  print(model.names)
  for box, cls, in zip(boxes, clss):
    x, y, w, h = box
    label = int(cls)
    x1, y1, x2, y2 = x-w/2, y-h/2, x+w/2, y+h/2
    annotator.box_label([x1, y1, x2, y2], label=classes[label] + ", Image Id: " + SYMBOL_MAP[classes[label]], color=(0,0,255))
  
  cv2.imwrite(f"/home/pi/MDP2/src/images/results/{filename}", annotator.im)
#  results.save(f"/home/pi/MDP2/src/images/results/{filename}")
  preds = []
  for c in results.boxes.cls:
    preds.append(classes[int(c)])
  return preds
   

def preprocess(path, new_width, new_height):
  with Image.open(path) as im:
    # Resize image
    width, height = im.size
    if height >= width:
        left = 0
        top = (height-width)/2
        right = width
        bottom = width+(height-width)/2
    else:
        left = (width-height)/2
        top = 0
        right = height+(width-height)/2
        bottom = height
    #im_cropped = im.crop((left, top, right, bottom))
    im_resized = np.asarray(im.resize((new_width, new_height), resample=Image.HAMMING))
    # Convert RGB to BGR
    #bgr_image = cv2.cvtColor(im_resized, cv2.COLOR_RGB2BGR)
    bgr_image = im_resized
    # Alter brightness
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_RGB2GRAY)
    cols, rows = gray.shape
    brightness = np.sum(bgr_image) / (255 * cols * rows)
    minimum_brightness = 0.8
    ratio = brightness / minimum_brightness

    if ratio < 1:
      bgr_image = cv2.convertScaleAbs(bgr_image, alpha = 1 / ratio, beta = 0)
      
    #convert from BGR to RGB
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    #filename = path[path.rindex('-')+1:]
    filename = path.split('/')[-1]
    #image_path = f"./images/{filename}"
    image_path = f"/home/pi/MDP2/src/images/preprocess/{filename}"
    cv2.imwrite(image_path, rgb_image)
    return image_path
  

  
