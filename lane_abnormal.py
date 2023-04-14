import cv2
import dlib
import math
import time
from twilio.rest import Client
import numpy as np
carCascade = cv2.CascadeClassifier('cars.xml')
video = cv2.VideoCapture('test/accident_3.mp4')

WIDTH = 1280
HEIGHT = 720
length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
total_frame = int(video.get(cv2.CAP_PROP_FPS))

frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

size = (int(frame_width),int(frame_height))
region_of_interest_vertices = [
    (0,frame_height),
    (frame_height / 2, frame_height / 2),
    (frame_height,frame_height),
]
def getFirstFrame(videofile):
    vidcap = cv2.VideoCapture(videofile)
    success, image = vidcap.read()
    if success:
        return image
        
left_avg=np.array([])
right_avg=np.array([])
img = getFirstFrame('test/accident_3.mp4')
lst =[]
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

kernel_size = 5
blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)

low_threshold = 50
high_threshold = 150
edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

rho = 1  
theta = np.pi / 180  
threshold = 100  
min_line_length = 10  
max_line_gap = 2000  
line_image = np.copy(img) * 0  

lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                        min_line_length, max_line_gap)

for line in lines:
    for x1,y1,x2,y2 in line:
        cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)
        lst.append([(x1,y1),(x2,y2)])
lines_edges = cv2.addWeighted(img, 0.8, line_image, 1, 0)

cv2.imwrite('testt.jpg',lines_edges)
print(lst)
def calculateDistance(x,y):
    for i in lst:
        distance = math.sqrt(math.pow(i[0][0] -x, 2) + math.pow(i[0][1] - y, 2))
        distance2 = math.sqrt(math.pow(i[1][0] - x, 2) + math.pow(i[1][0] - y, 2))

        if distance <= 1 or distance <= 1:
            return True
        
def estimateSpeed(loc1, loc2):
    distance = math.sqrt(math.pow(loc2[0] - loc1[0], 2) + math.pow(loc2[1] - loc1[1], 2))
    ppm = 8.8
    d_meters = distance / ppm
    fps = 100
    speed = d_meters * fps * 3.6 
    return speed

def estimateanomal(loc1, loc2):
    distance = abs(loc2[0] - loc1[0]) * 10
    ppm = 8.8
    d_meters = distance / ppm 
    return d_meters  


def trackMultipleObjects():
    cannyed_image=None
    rectangleColor = (0, 255, 0)
    frameCounter = 0
    currentCarID = 0
    fps = 0
    account_sid = "AC772840fd81ec571de07148a1e2e1a1d0"
    auth_token  = "34ad3b362b9da7a46eb66c491861f2e8"
    client = Client(account_sid, auth_token)
    lines =[]
    warn=False
    a = None
    trackCar = {}
    carNumbers = {}
    carLocation1 = {}
    carLocation2 = {}
    abnormal_cars={}
    speed_cars={}
    speed = [None] * 1000
    anomal = [None] * 1000
    fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
    laneCounter = {}
    out = cv2.VideoWriter('output/accident_1_0101.mp4',fourcc, 20, size)


    while frameCounter<5396:
        rc, image = video.read()
        if rc == False:
            break
        if type(image) == type(None):
            break
        
        resultImage = image.copy()
        
        frameCounter = frameCounter + 1
        
        carIDdelete = []

        for carID in trackCar.keys():
            trackingQuality = trackCar[carID].update(image)
            
            if trackingQuality < 5:
                carIDdelete.append(carID)
                
        for carID in carIDdelete:
            trackCar.pop(carID, None)
            carLocation1.pop(carID, None)
            carLocation2.pop(carID, None)
        
        if not (frameCounter % 10):
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray,(5,5),0)
            dilated = cv2.dilate(blur,np.ones((3,3)))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            closing = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel) 
            cars = carCascade.detectMultiScale(dilated, 1.1, 1)

            for (_x, _y, _w, _h) in cars:
                x = int(_x)
                y = int(_y)
                w = int(_w)
                h = int(_h)
            
                x_bar = x + 0.5 * w
                y_bar = y + 0.5 * h
                
                matchCarID = None
            
                for carID in trackCar.keys():
                    trackedPosition = trackCar[carID].get_position() 
                    
                    t_x = int(trackedPosition.left())
                    t_y = int(trackedPosition.top())
                    t_w = int(trackedPosition.width())
                    t_h = int(trackedPosition.height())
                    
                    t_x_bar = t_x + 0.5 * t_w
                    t_y_bar = t_y + 0.5 * t_h
                
                    if ((t_x <= x_bar <= (t_x + t_w)) and (t_y <= y_bar <= (t_y + t_h)) and (x <= t_x_bar <= (x + w)) and (y <= t_y_bar <= (y + h))):
                        matchCarID = carID
                        if calculateDistance(carLocation1[matchCarID][0],carLocation1[matchCarID][1]) and carLocation1[matchCarID][5]==0:
                            print("Aleart!",carID)
                            carLocation1[carID][4] +=1
                            carLocation1[carID][5] =1
                if matchCarID is None:
                    tracker = dlib.correlation_tracker()
                    tracker.start_track(image, dlib.rectangle(x, y, x + w, y + h)) 
                   
                    trackCar[currentCarID] = tracker
                    carLocation1[currentCarID] = [x, y, w, h,0,0]
                    currentCarID = currentCarID + 1


    
        for i in carLocation1.keys():
            if frameCounter % 1 == 0:
                [x1, y1, w1, h1,c,a] = carLocation1[i]
                
            


        

  
        cv2.imshow('result', resultImage)
        out.write(resultImage)
        if cv2.waitKey(33) == 27:
            break
    video.release()
    cv2.destroyAllWindows()
    out.release()
    
if __name__ == '__main__':
    trackMultipleObjects()
