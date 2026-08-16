"""
registration.py
Регистрация: config.NUM_REGISTRATION_PHOTOS фото лица через камеру,
embedding каждого, сохранение в SQLite.
"""
import cv2

from core import face_detector, face_embedder
from core.camera_stream import CameraStream
from database import db_manager
import config


def register_user(name):
    """
    'c' — снимок (только если лицо найдено в кадре), 'q' — выход.
    Возвращает True, если собраны все config.NUM_REGISTRATION_PHOTOS фото.
    """
    user_id = db_manager.add_user(name)
    photos_taken = 0

    with CameraStream() as cam:
        print(f"Регистрация '{name}': нужно {config.NUM_REGISTRATION_PHOTOS} фото.")
        print("'c' — снимок, 'q' — выход.")

        while photos_taken < config.NUM_REGISTRATION_PHOTOS:
            frame = cam.read_frame()
            if frame is None:
                break

            detection = face_detector.detect_face(frame)
            display = frame.copy()

            if detection is not None:
                x, y, w, h = detection["box"]
                cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
                status = "Лицо найдено - 'c' для снимка"
            else:
                status = "Лицо не найдено"

            cv2.putText(display, f"{status} | {photos_taken}/{config.NUM_REGISTRATION_PHOTOS}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.imshow("Registration - c: capture, q: quit", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("c") and detection is not None:
                face = face_detector.extract_face(frame, detection)
                embedding = face_embedder.get_embedding(face)
                db_manager.add_embedding(user_id, embedding)
                photos_taken += 1
                print(f"Снимок {photos_taken}/{config.NUM_REGISTRATION_PHOTOS} сохранён.")

    cv2.destroyAllWindows()

    success = photos_taken == config.NUM_REGISTRATION_PHOTOS
    print("Регистрация завершена." if success else
          f"Регистрация прервана ({photos_taken}/{config.NUM_REGISTRATION_PHOTOS}).")
    return success
