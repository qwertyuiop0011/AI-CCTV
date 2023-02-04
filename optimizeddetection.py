import cv2
import numpy as np

#Web Camera
cap = cv2.VideoCapture('test/accident_1.mp4')

min_width_rectangle = 80
min_height_rectangle = 80

count_line_position = 550
# Initialize Substructor
algo = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)



def center_handle(x,y,w,h):
    x1=int(w/2)
    y1=int(h/2)
    cx=x+x1
    cy=y+y1
    return cx,cy

detect = []
offset = 6
counter = 0

while True:
    ret, video = cap.read()
    gray = cv2.cvtColor(video, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 5)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))

    vi_sub = algo.apply(blur)
    car = cv2.dilate(vi_sub, np.ones((5,5)))
    cars = cv2.morphologyEx(car, cv2.MORPH_CLOSE, kernel)
    cars = cv2.morphologyEx(cars, cv2.MORPH_CLOSE, kernel)
    countersahpe, h = cv2.findContours(cars, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


    for (i, c) in enumerate(countersahpe):
        (x,y,w,h) = cv2.boundingRect(c)
        val_counter = (w>=min_width_rectangle) and (h>= min_height_rectangle)
        if not val_counter:
            continue
        cv2.rectangle(video,(x,y),(x+w,y+h),(0,255,0),2)


        center = center_handle(x,y,w,h)
        detect.append(center)

        for (x,y) in detect:
            if y<(count_line_position + offset) and  y>(count_line_position - offset):
                counter+=1
                detect.remove((x,y))

    cv2.imshow('Detector',video)


    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()