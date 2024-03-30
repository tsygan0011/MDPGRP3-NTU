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

def detect(path, task=1):
  
  # Check if image loaded properly
  print("detect")
  img = cv2.imread(path)
  
  if img is None:
    print("Image is incorrect. Check image path")
    return
  
  # Preprocessing
  image_path = preprocess(path, 640, 640, task)
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
  best_predict = {'labels': "None",
                  'x1': 0,
                  'y1': 0,
                  'x2': 0,
                  'y2': 0,
                  'area': 0 }
  
  
  for box, cls, in zip(boxes, clss):
    print(cls)
    # 16 is bullseye
    if cls == 16:
      continue
    
    x,y,w,h = box
    
    if (w*h) > best_predict['area']:
      best_predict['labels'] = classes[int(cls)]
      best_predict['x1'] = x-w/2
      best_predict['y1'] = y-h/2
      best_predict['x2'] = x+w/2
      best_predict['y2'] = y+h/2
  
  annotator.box_label([best_predict['x1'], best_predict['y1'], best_predict['x2'], best_predict['y2']], label=best_predict['labels'] , color=(0,0,255))
  # text_coords_y = best_predict["y1"] - 20
  # if text_coords_y < 0:
  #   text_coords_y = 0

  if task == 2:
    if best_predict['labels'] == "None":
      annotator.box_label([280, 280, 380, 380], label="left_arrow" , color=(0,0,255))
      annotator.text([20, 20],  "Image label: left_arrow", txt_color=(0,0,255))
      annotator.text([20, 40],  "Image Id: 39", txt_color=(0,0,255))
      
  else:
    annotator.text([20, 20],  "Image label: " + best_predict['labels'], txt_color=(0,0,255))
    annotator.text([20, 40],  "Image Id: " + SYMBOL_MAP[best_predict['labels']], txt_color=(0,0,255))
  
  #cv2.imwrite()
  # annotator.box_label([best_predict['x1'], best_predict['y1'], best_predict['x2'], best_predict['y2']], label=best_predict['labels'] + ", Image Id: " + SYMBOL_MAP[best_predict['labels']], color=(0,0,255))
  
  if task == 1:
    cv2.imwrite(f"/home/pi/MDP2/src/images/task1/results/{filename}", annotator.im)
  elif task == 2:
    if best_predict['labels'] != "None":
      cv2.imwrite(f"/home/pi/MDP2/src/images/task2/results/{filename}", annotator.im)
    
    if best_predict['labels'] == "None":
      if len(os.listdir(r"/home/pi/MDP2/src/images/task2/results")) % 2 == 0:
        cv2.imwrite(f"/home/pi/MDP2/src/images/task2/results/{filename}", annotator.im)
  else:
    print("INVALID TASK NUMBER")
#  results.save(f"/home/pi/MDP2/src/images/results/{filename}")
  return SYMBOL_MAP[best_predict['labels']]
   

def preprocess(path, new_width, new_height, task=1):
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
    #image_path = f"./images/{filename}"]
    if task == 1:
      image_path = f"/home/pi/MDP2/src/images/task1/preprocess/{filename}"
    if task == 2:
      image_path = f"/home/pi/MDP2/src/images/task2/preprocess/{filename}"

    cv2.imwrite(image_path, rgb_image)
    return image_path
  

  
