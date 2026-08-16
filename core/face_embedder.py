"""
face_embedder.py
Превращает изображение лица (config.FACE_SIZE) в embedding-вектор (512,)
через pretrained FaceNet (keras-facenet). L2-нормализация — часть архитектуры
модели (последний слой), препроцессинг выполняется внутри embedder.embeddings().
"""

import numpy as np
from keras_facenet import FaceNet

embedder = FaceNet()


def get_embedding(face_image):
    """
    face_image: numpy array config.FACE_SIZE, уже вырезанное лицо.
    Возвращает numpy array формы (512,).
    """
    face_batch = np.expand_dims(face_image, axis=0)
    embeddings = embedder.embeddings(face_batch)
    return embeddings[0]
