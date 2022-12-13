import cv2
import dlib
import math
from twilio.rest import Client

carCascade = cv2.CascadeClassifier('cars.xml')
video = cv2.VideoCapture('accident_3_5.mp4')

WIDTH = 1280
HEIGHT = 720
length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
total_frame = int(video.get(cv2.CAP_PROP_FPS))

frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

size = (int(frame_width),int(frame_height))
def region_of_interest(img, vertices, color3=(255,255,255), color1=255): 

    mask = np.zeros_like(img) 
    
    if len(img.shape) > 2: 
        color = color3
    else: 
        color = color1
        
    cv2.fillPoly(mask, vertices, color)
    
    ROI_image = cv2.bitwise_and(img, mask)
    return ROI_image

def mark_img(img, blue_threshold=200, green_threshold=200, red_threshold=200): # 흰색 차선 찾기

    bgr_threshold = [blue_threshold, green_threshold, red_threshold]

    thresholds = (image[:,:,0] < bgr_threshold[0]) \
                | (image[:,:,1] < bgr_threshold[1]) \
                | (image[:,:,2] < bgr_threshold[2])
    mark[thresholds] = [0,0,0]
    return mark

cap = cv2.VideoCapture('accident_3_5.mp4') 

while(cap.isOpened()):
    ret, image = cap.read()
    height, width = image.shape[:2] 
    vertices = np.array([[(0,0),(0, height), (width,height),(width,0)]], dtype=np.int32)

    roi_img = region_of_interest(image, vertices, (0,0,255)) 

    mark = np.copy(roi_img) 
    mark = mark_img(image) 

    color_thresholds = (mark[:,:,0] == 0) & (mark[:,:,1] == 0) & (mark[:,:,2] > 200)
    image[color_thresholds] = [0,0,255]

    cv2.imshow('results',image) 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

def estimateSpeed(loc1, loc2):
    distance = math.sqrt(math.pow(loc2[0] - loc1[0], 2) + math.pow(loc2[1] - loc1[1], 2))
    ppm = 8.8
    d_meters = distance / ppm
    fps = 100
    speed = d_meters * fps * 3.6 
    return speed

def estimateanomal(loc1, loc2):
    distance = abs(loc2[0] - loc1[0]) * 10
    line_count = 1
    
    ppm = 8.8
    d_meters = distance / ppm 
    return d_meters  

def trackMultipleObjects():
    rectangleColor = (0, 255, 0)
    frameCounter = 0
    currentCarID = 0
    fps = 0


    warn=False
    trackCar = {}
    carNumbers = {}
    carLocation1 = {}
    carLocation2 = {}
    abnormal_cars={}
    speed_cars={}
    speed = [None] * 1000
    anomal = [None] * 1000
    fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
    
    out = cv2.VideoWriter('output/norm.mp4',fourcc, 20, size)


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
            cars = carCascade.detectMultiScale(image, 1.1, 1)
            
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
                
                if matchCarID is None:
                    tracker = dlib.correlation_tracker()
                    tracker.start_track(image, dlib.rectangle(x, y, x + w, y + h)) 
                    
                    trackCar[currentCarID] = tracker
                    carLocation1[currentCarID] = [x, y, w, h]

                    currentCarID = currentCarID + 1

        for carID in trackCar.keys():
            trackedPosition = trackCar[carID].get_position()
                    
            t_x = int(trackedPosition.left())
            t_y = int(trackedPosition.top())
            t_w = int(trackedPosition.width())
            t_h = int(trackedPosition.height())

            if (carID in abnormal_cars.keys()) or (carID in speed_cars.keys()):
                cv2.rectangle(resultImage, (t_x, t_y), (t_x + t_w, t_y + t_h), (0,0,255), 4)
            else:
                cv2.rectangle(resultImage, (t_x, t_y), (t_x + t_w, t_y + t_h), rectangleColor, 4)

            carLocation2[carID] = [t_x, t_y, t_w, t_h]
        
        for i in carLocation1.keys():
            if frameCounter % 1 == 0:
                [x1, y1, w1, h1] = carLocation1[i]
                [x2, y2, w2, h2] = carLocation2[i]
        
                carLocation1[i] = [x2, y2, w2, h2]

                if [x1, y1, w1, h1] != [x2, y2, w2, h2]:
                    if (speed[i] == None or speed[i] == 0) and y1 >= 275 and y1 <= 285:
                        speed[i] = estimateSpeed([x1, y1, w1, h1], [x2, y2, w2, h2])
                    anomal[i] = estimateanomal([x1, y1, w1, h1], [x2, y2, w2, h2])
                    
                    if speed[i] != None:
                        if speed[i]>= 250 and warn== False:
                            speed_cars[i]=1


                            warn = True

                            cv2.putText(resultImage, "Speed Violation", (int(x1 + w1/2), int(y1-5)),cv2.FONT_HERSHEY_SIMPLEX,0.75, (0,0,255), 2)
                    if anomal[i] != None:
                        if anomal[i] >= 15 and warn== False:
                            abnormal_cars[i] = 1


                            warn = True

                            cv2.putText(resultImage, "High Anomaly", (int(x1 + w1/2), int(y1+15)),cv2.FONT_HERSHEY_SIMPLEX,0.75, (0,0,255), 2)

        cv2.imshow('result', resultImage)
        out.write(resultImage)
        if cv2.waitKey(33) == 27:
            break
        
    video.release()
    cv2.destroyAllWindows()
    out.release()

if __name__ == '__main__':
    trackMultipleObjects()
