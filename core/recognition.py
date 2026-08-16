"""
recognition.py
Единый identify() pipeline: detect -> extract -> embed -> match.
Используется и в live-камере, и в распознавании по файлу, и в evaluation.py.
"""
import cv2

from core import face_detector, face_embedder, matcher
from core.camera_stream import CameraStream
from database import db_manager


def identify(frame):
    """
    1:N identification для одного кадра.
    Возвращает None, если лицо не найдено, иначе:
        {"user_id", "name", "similarity", "box"}
    """
    detection = face_detector.detect_face(frame)
    if detection is None:
        return None

    face = face_detector.extract_face(frame, detection)
    embedding = face_embedder.get_embedding(face)
    user_id, name, similarity = matcher.find_match(embedding)

    return {
        "user_id": user_id,
        "name": name,
        "similarity": similarity,
        "box": detection["box"],
    }


def recognize_from_image(image_path):
    """1:N identification на одном изображении из файла."""
    frame = cv2.imread(image_path)
    if frame is None:
        raise FileNotFoundError(f"Не удалось прочитать изображение: {image_path}")
    return identify(frame)


def recognize_from_camera():
    """Live-распознавание через камеру с отметкой посещаемости. 'q' — выход."""
    with CameraStream() as cam:
        print("Распознавание запущено. Нажмите 'q' для выхода.")
        marked_this_session = set()

        while True:
            frame = cam.read_frame()
            if frame is None:
                break

            result = identify(frame)

            if result is not None:
                x, y, w, h = result["box"]
                is_known = result["name"] != "Unknown"
                color = (0, 255, 0) if is_known else (0, 0, 255)
                label = f'{result["name"]} ({result["similarity"]:.2f})'

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, max(0, y - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                if result["user_id"] is not None and result["user_id"] not in marked_this_session:
                    if db_manager.mark_attendance(result["user_id"]):
                        print(f'Отмечено посещение: {result["name"]}')
                    marked_this_session.add(result["user_id"])

            cv2.imshow("Recognition - press q to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()
