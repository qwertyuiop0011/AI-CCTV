# AI-CCTV
AI-CCTV is a video analyzing tool that uses OpenCV and CascadeClassifiers to detect any on-road hazardous situations and report them to nearby vehicles.

## How it works
AI-CCTV utilizes OpenCV to locate cars within a video and draws a retangle around each vehicle. The location process is done through CascadeClassifiers, which performs most ideally (speed + accuracy) in a constrained situtations like when low-quality surveillance videos are provided. The program tracks the coordinates of these rectangles frame-by-frame and raises a signal whenever a rectangle is detected to move anomaly (threshold parameters can be customized depending on road length and curvature). This function is implemented as `trackMultipleObjects()` in `cardetect_with_exp.py`. Whenever an anomaly signal is raised, the program alerts all drivers within the camera's vicinity through TwilioAPI. This step is currently incomplete and in need of updates.

## Caveats
AI-CCTV is targeted to maximize its performance in unfavored situations like when videos are in low-quality or when vehicles are moving unexpectedly. Besides its detection accuracy, it also focuses on the speed of detection as it is imperative to alert drivers quickly before they get into an accident and the cost of such programs as it aims to be applicable to any on-road surveillance cameras.
