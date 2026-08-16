"""
camera_stream.py
Единая точка доступа к веб-камере (context manager гарантирует release()).
"""
import cv2


class CameraStream:
    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.cap = None

    def start(self):
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Не удалось открыть камеру (index={self.camera_index})")
        return self

    def read_frame(self):
        if self.cap is None:
            raise RuntimeError("Камера не запущена. Сначала вызовите start().")
        ret, frame = self.cap.read()
        return frame if ret else None

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
