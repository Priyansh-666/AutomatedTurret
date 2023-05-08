import torch
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pyfirmata


cap = cv2.VideoCapture(0)
ws, hs = 1920, 1080
cap.set(3, ws)
cap.set(4, hs)

servoPos = [90, 90]
# Loading in yolov5s - you can switch to larger models such as yolov5m or yolov5l, or smaller such as yolov5n
model = torch.hub.load('ultralytics/yolov5', 'yolov5l')
while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    results = model(img)
    label_map = {
    0: 'person',
    1: 'bicycle',
    2: 'car',
    3: 'motorcycle',
    4: 'airplane',
    5: 'bus',
    6: 'train',
    7: 'truck',
    8: 'boat',
    9: 'traffic light',
    10: 'fire hydrant',
    11: 'stop sign',
    12: 'parking meter',
    13: 'bench',
    14: 'bird',
    15: 'cat',
    16: 'dog',
    17: 'horse',
    18: 'sheep',
    19: 'cow',
    20: 'elephant',
    21: 'bear',
    22: 'zebra',
    23: 'giraffe',
    24: 'backpack',
    25: 'umbrella',
    26: 'handbag',
    27: 'tie',
    28: 'suitcase',
    29: 'frisbee',
    30: 'skis',
    31: 'snowboard',
    32: 'sports ball',
    33: 'kite',
    34: 'baseball bat',
    35: 'baseball glove',
    36: 'skateboard',
    37: 'surfboard',
    38: 'tennis racket',
    39: 'bottle',
    40: 'wine glass',
    41: 'cup',
    42: 'fork',
    43: 'knife',
    44: 'spoon',
    45: 'bowl',
    46: 'banana',
    47: 'apple',
    48: 'sandwich',
    49: 'orange',
    50: 'broccoli',
    51: 'carrot',
    52: 'hot dog',
    53: 'pizza',
    54: 'donut',
    55: 'cake',
    56: 'chair',
    57: 'couch',
    58: 'potted plant',
    59: 'bed',
    60: 'dining table',
    61: 'toilet',
    62: 'tv',
    63: 'laptop',
    64: 'mouse',
    65: 'remote',
    66: 'keyboard',
    67: 'cell phone',
    68: 'microwave',
    69: 'oven',
    70: 'toaster',
    71: 'sink',
    72: 'refrigerator',
    73: 'book',
    74: 'clock',
    75: 'vase',
    76: 'scissors',
    77: 'teddy bear',
    78: 'hair drier',
    79: 'toothbrush'
}
    results = model(img)
    boxes = results.xyxy[0].tolist()
    labels = results.xyxy[0][:, -1].tolist()
    label_names = [label_map[label] for label in labels]

    class_name = 0
    class_index = list(model.names.keys()).index(class_name)
    class_results = results.pred[class_index]

    largest_obj_index = 0
    largest_obj_size = 0
    for i, obj in enumerate(class_results):
        obj_size = (obj[2] - obj[0]) * (obj[3] - obj[1])
        if obj_size > largest_obj_size:
            largest_obj_size = obj_size
            largest_obj_index = i

# Get the bounding box coordinates of the largest object
    try:
        largest_obj_box = class_results[largest_obj_index][:4].tolist()

        # Crop the image to only include the largest object
        x1, y1, x2, y2 = largest_obj_box
                # print("person")
                # print(x1,y1,x2,y2)
                # print(servoX)

        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(img, 'person', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("Image", img)
        if cv2.waitKey(10)&0xFF== ord('q'):
            break
    except:
        continue