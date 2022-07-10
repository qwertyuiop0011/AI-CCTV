import os
from PIL import Image
import imageio
import numpy as np
import cv2 as cv

fgbg = cv.createBackgroundSubtractorMOG2()
count = 0
for path in os.listdir("/Users/leejeesung/Downloads/normal/000000"):
    if not path.startswith('.'):
        frame = Image.open("/Users/leejeesung/Downloads/normal/000000/"+path)
        frame = np.array(frame)
        avg1 = np.float32(frame)
        avg2 = np.float32(frame)
        cv.accumulateWeighted(frame,avg1,0.1)
        cv.accumulateWeighted(frame,avg2,0.01)

        res1 = cv.convertScaleAbs(avg1)
        res2 = cv.convertScaleAbs(avg2)

        #cv.imshow('avg1',res1)
        cv.imshow('avg2',res2)
        keyboard = cv.waitKey(30)
        
        if keyboard == 'q' or keyboard == 27:
            print("CHECK",keyboard)
            break

cv.destroyAllWindows()
