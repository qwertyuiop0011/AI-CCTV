import tensorflow as tf
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from tf2_yolov4.anchors import YOLOV4_ANCHORS
from tf2_yolov4.model import YOLOv4
import cv2
import numpy as np

def load_image(file):
    img = tf.io.read_file(file)
    img = tf.io.decode_image(img)
    return img

def resize_image(img):
    img = tf.image.resize(img, (HEIGHT, WIDTH))
    img = tf.expand_dims(img, axis=0)/255
    return images

def load_yolov4_model():
    model = YOLOv4(
        input_shape=(HEIGHT, WIDTH, 3),
        anchors=YOLOV4_ANCHORS,
        num_classes=80,
        training=False,
        yolo_max_boxes=20,
        yolo_iou_threshold=0.5,
        yolo_score_threshold=0.73,
    )
    model.load_weights('yolov4.h5')
    return model

def detected_photo(boxes, scores, classes, detections,image):
    boxes = (boxes[0] * [WIDTH, HEIGHT, WIDTH, HEIGHT]).astype(int)
    scores = scores[0]
    classes = classes[0].astype(int)
    detections = detections[0]

    CLASSES = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
        'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
        'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
        'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
        'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
        'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
        'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
        'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
    image_cv = image.numpy()

    for (xmin, ymin, xmax, ymax), score, class_idx in zip(boxes, scores, classes):

        if score > 0:
            if class_idx == 2: 
                cv2.rectangle(image_cv, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0,255,0), thickness= 2)
                text = CLASSES[class_idx] + ': {0:.2f}'.format(score)
                cv2.putText(image_cv, text, (int(xmin), int(ymin) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    return image_cv

def proccess_frame(photo, model):
    images = resize_image(photo)
    boxes, scores, classes, detections = model.predict(images)
    result_img = detected_photo(boxes, scores, classes, detections,images[0])
    return result_img


def detect_car(input_video_name, output_video_name, frames_to_save = 50):

    model = load_yolov4_model()

    my_video = cv2.VideoCapture(input_video_name)

    out = cv2.VideoWriter(output_video_name, cv2.VideoWriter_fourcc(*'MJPG'), 30, (WIDTH ,HEIGHT))

    success = 1
    i = 0
    while success and i < frames_to_save: 
        success, image = my_video.read() 
        if success:
            result_img = proccess_frame(tf.convert_to_tensor(image), model)   

            out.write((result_img*255).astype('uint8'))                                           
            i = i + 1
            print(i)
    out.release()                                         


if __name__ == "__main__":

    WIDTH, HEIGHT = (1024, 768)
    detect_car(input_video_name='test/normal.mp4', output_video_name ='output/normalv3.mp4', frames_to_save = 20)