import torch
import cv2
from PIL import Image
import numpy as np
import pathlib
pathlib.PosixPath = pathlib.WindowsPath

DEVICE = "cuda" if torch.cuda.is_available() else "cpu" 
  
weights_path = "best.pt"  # Cambia esto a la ruta de tu modelo
model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path)

cam = cv2.VideoCapture(0)
  
while(True): 
    ret, frame = cam.read()
    frame = frame[:, :, [2,1,0]]
    frame = Image.fromarray(frame) 
    frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)

    results = model(frame,size=640)
    cv2.imshow('YOLO', np.squeeze(results.render()))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
cam.release()
cv2.destroyAllWindows()