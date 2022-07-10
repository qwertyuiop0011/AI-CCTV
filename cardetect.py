import time
import cv2 as cv
import numpy as np
import math

prototxt_path = "MobileNetSSD_deploy.prototxt"
model_path = "MobileNetSSD_deploy.caffemodel"

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"]

net = cv.dnn.readNetFromCaffe(prototxt_path, model_path)

def process_frame(next_frame):
    print("Framing")
    rgb = cv.cvtColor(next_frame, cv.COLOR_BGR2RGB)
    (H, W) = next_frame.shape[:2]

    blob = cv.dnn.blobFromImage(next_frame, size=(300, 300), ddepth=cv.CV_8U)
    net.setInput(blob, scalefactor=1.0/127.5, mean=[127.5, 127.5, 127.5])
    detections = net.forward()

    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.7:
            idx = int(detections[0, 0, i, 1])
            if CLASSES[idx] != "car" or CLASSES[idx] != "bus" or CLASSES[idx] != "train" or CLASSES[idx] != "bicyle" or CLASSES[idx] != "motorbike":
                continue
            box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
            (startX, startY, endX, endY) = box.astype("int")
            
            cv.rectangle(next_frame, (startX, startY), (endX, endY), (0, 255, 0), 3)

    return next_frame


def VehicheDetection(filename):
    print("VechicleDection_MobileNetSSD")
    cap = cv.VideoCapture(filename)

    frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    fps = 20
    size = (int(frame_width),int(frame_height))
    fourcc = cv.VideoWriter_fourcc('m','p','4','v')
    out = cv.VideoWriter()
    success = out.open('output_mobilenetssd.mov', fourcc, fps, size, True)
    frame_count = 0
    # start timer
    t1 = time.time()
    while True:
        ret, next_frame = cap.read()
        
        if ret == False: break

        frame_count += 1
        next_frame = process_frame(next_frame)

        out.write(next_frame)
        
        key = cv.waitKey(50)
        
        if key == 27:
            break

    t2 = time.time()

    fps = str( float(frame_count / float(t2 - t1))) + ' FPS'

    cap.release()
    cv.destroyAllWindows()
    out.release()

VehicheDetection("video.mp4")
