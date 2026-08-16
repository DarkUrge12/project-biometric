"""
face_detector.py
Обнаружение лица на изображении с помощью MTCNN (pretrained, без дообучения).
"""

import cv2
from mtcnn import MTCNN

import config

detector = MTCNN()


def detect_face(image):
    """
    image: numpy array (BGR, как отдаёт OpenCV)
    Возвращает словарь {"box", "confidence", "keypoints"} или None, если лицо не найдено.
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    detections = detector.detect_faces(image_rgb)

    if len(detections) == 0:
        return None

    # Если несколько лиц — берём то, в котором модель наиболее уверена.
    best_detection = max(detections, key=lambda d: d["confidence"])
    return best_detection


def extract_face(image, detection):
    """
    Вырезает лицо по bounding box и ресайзит до config.FACE_SIZE.
    Границы box клиппятся по размеру изображения (MTCNN может вернуть
    box, частично выходящий за пределы кадра у самого края).
    """
    x, y, width, height = detection["box"]
    x, y = max(0, x), max(0, y)

    img_h, img_w = image.shape[:2]
    x2 = min(x + width, img_w)
    y2 = min(y + height, img_h)

    face = image[y:y2, x:x2]
    face_resized = cv2.resize(face, config.FACE_SIZE)
    return face_resized
