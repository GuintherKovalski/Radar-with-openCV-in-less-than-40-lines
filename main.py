import cv2 ## OPENCV E pillow
import numpy as np
import time
from PIL import Image
pt = np.array([0,0]) 
cap = cv2.VideoCapture('carros_3.mp4') #aqu pega a imagem do webcam
global Iter
img = Image.open('carro_teste.png').convert('L')
img.save('greyscale2.png')
Iter = 0
dist = [] 
while True:
    start = time.time()
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    img_rgb = frame #cv2.imread('laranjaGrande.png')
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    template = cv2.imread('greyscale2.png',0)
    w, h = template.shape[::-1]
    res = cv2.matchTemplate(gray,template,cv2.TM_CCOEFF_NORMED)
    threshold = 0.5 
    loc = np.where( res >= threshold)
    pt1 = pt
    Iter = Iter+1
    for pt in zip(*loc[::-1]):
        cv2.rectangle(gray, pt, (pt[0] + w, pt[1] + h), (0,0,255), 3) 
    '''based in the car size (Celta) and in the video screen size (640 pixels of width)
    we calculated that each pixel is equivalent to approximately  0.03 Meters, 
    you need to calculate this for your case'''
    dist.append(((pt1[0]-pt[0])**2+(pt1[1]-pt[1])**2)**0.5*((0.03)))            
    cv2.imshow('OCR',gray) 
    end = time.time()   
    if Iter == 30:#in this case, 30 is the nu number of FPS in the video used
        print('Speed is:',sum(np.array(dist))*(3.6), 'Km/h')
        dist = []
        Iter = 0    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break 
cap.release()
cv2.destroyAllWindows()

