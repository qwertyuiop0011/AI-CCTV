# AI-CCTV
AI-CCTV is a video analyzing tool that uses OpenCV and CascadeClassifiers to detect any on-road hazardous situations and report them to nearby vehicles.

## How it works
AI-CCTV utilizes OpenCV to locate cars within a video and draws a retangle around each vehicle. The location process is done through CascadeClassifiers, which performs most ideally (speed + accuracy) in a constrained situtations like when low-quality surveillance videos are provided. The program tracks the coordinates of these rectangles frame-by-frame and raises a signal whenever a rectangle is detected to move anomaly (threshold parameters can be customized depending on road length and curvature).
