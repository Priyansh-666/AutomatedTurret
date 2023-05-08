import torch
import cv2
from flask import Flask, render_template, Response
import numpy as np
from openport import PortOpen


app = Flask(__name__)

cap = cv2.VideoCapture(0)
ws, hs = 1920, 1080
cap.set(3, ws)
cap.set(4, hs)


servoPos = [90, 90]


model = torch.hub.load('ultralytics/yolov5', 'yolov5l')
label_map = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
             5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
             10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
             14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
             20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack',
             25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee',
             30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat',
             35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket',
             39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon',
             45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli',
             51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair',
             57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet',
             62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone',
             68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book',
             74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}

def detect_objects(img):
    initialize = PortOpen()
    results = model(img)
    boxes = results.xyxy[0].tolist()
    labels = results.xyxy[0][:, -1].tolist()
    label_names = [label_map[label] for label in labels]
    # servoX = 0
    for box, label in zip(boxes, label_names):
        x1, y1, x2, y2 = map(int, box[:4])
        if x1 < 0 or y1 < 0 or x2 >= img.shape[1] or y2 >= img.shape[0]:
            continue
        if(label == "person"):
            servoX = np.interp(x1, [0, ws], [20, 160])
            fx, fy = x1,y1
            pos = [fx, fy]

            #convert coordinat to servo degree
            servoX = np.interp(fx, [0, ws], [20, 160])
            servoY = np.interp(fy, [0, hs], [20, 160])

            if servoX < 0:
                servoX = 0
            elif servoX > 180:
                servoX = 180
            if servoY < 0:
                servoY = 0
            elif servoY > 180:
                servoY = 180
            
            
            servoPos[0] = servoX
            servoPos[1] = servoY
            initialize.move_servo(servoPos[0])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return img

def gen_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # Flip the frame horizontally
            frame = cv2.flip(frame, 1)
            
            frame = detect_objects(frame)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':

    app.run(debug=True,port=4001,threaded=True)
