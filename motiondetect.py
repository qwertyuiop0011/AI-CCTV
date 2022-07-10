import numpy as np
import cv2

from matplotlib import pyplot as plt

np.random.seed(31)

def fixColor(image):
    return(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


video_stream = cv2.VideoCapture('test.mp4')

frameIds = video_stream.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=30)

frames = []
for fid in frameIds:
    video_stream.set(cv2.CAP_PROP_POS_FRAMES, fid)
    ret, frame = video_stream.read()
    frames.append(frame)
    
video_stream.release()


medianFrame = np.median(frames, axis=0).astype(dtype=np.uint8)

sample_frame=frames[0]

avgFrame = np.average(frames, axis=0).astype(dtype=np.uint8)
sample_frame=frames[0]
grayMedianFrame = cv2.cvtColor(medianFrame, cv2.COLOR_BGR2GRAY)

graySample=cv2.cvtColor(sample_frame, cv2.COLOR_BGR2GRAY)

dframe = cv2.absdiff(graySample, grayMedianFrame)

blurred = cv2.GaussianBlur(dframe, (11,11), 0)
ret, tframe= cv2.threshold(blurred,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

(cnts, _) = cv2.findContours(tframe.copy(), cv2.RETR_EXTERNAL, 
                             cv2 .CHAIN_APPROX_SIMPLE)

for cnt in cnts:
    x,y,w,h = cv2.boundingRect(cnt)
    if y > 200: 
        cv2.rectangle(sample_frame,(x,y),(x+w,y+h),(0,255,0),2)

writer = cv2.VideoWriter("output2.mp4", 
                         cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 30,(640,480))

video_stream = cv2.VideoCapture('video.mp4')
total_frames=video_stream.get(cv2.CAP_PROP_FRAME_COUNT)


frameCnt=0
while(frameCnt < total_frames-1):

    frameCnt+=1
    ret, frame = video_stream.read()

    gframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    dframe = cv2.absdiff(gframe, grayMedianFrame)
    blurred = cv2.GaussianBlur(dframe, (11, 11), 0)
    ret, tframe= cv2.threshold(blurred,0,255,
                               cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    (cnts, _) = cv2.findContours(tframe.copy(), 
                                 cv2.RETR_EXTERNAL, cv2 .CHAIN_APPROX_SIMPLE)
    for cnt in cnts:
        x,y,w,h = cv2.boundingRect(cnt)
        if y > 200: 
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

    writer.write(cv2.resize(frame, (640,480)))
 
video_stream.release()
writer.release()