# Face Recognition Attendance System

Учебный проект: MTCNN (детекция) + FaceNet/keras-facenet (512-D embedding) +
cosine similarity (1:N identification с reject option) + SQLite (users, embeddings, attendance).

## Установка

Требуется Python 3.10+.

**1. Системная зависимость для GUI (Ubuntu/Debian) — обязательна, по умолчанию НЕ установлена:**
```bash
sudo apt install python3-tk
```
Без этого пакета `gui/app.py` не запустится (`ModuleNotFoundError: No module named 'tkinter'`).
CLI (`main.py`) и `evaluation.py` от этого не зависят.

**2. Python-зависимости:**
```bash
pip install -r requirements.txt
```
Проверено на: tensorflow 2.21.0, keras 3.15.1, mtcnn 1.0.0, keras-facenet 0.3.2, Python 3.12.
При первом запуске (`face_embedder.py` / `face_detector.py`) библиотеки автоматически
скачивают веса предобученных моделей — нужен интернет при первом запуске.

## Запуск

CLI:
```bash
python main.py
```

GUI (Tkinter):
```bash
python gui/app.py
```

## Оценка (FAR / FRR / EER / ROC / DET)

1. Зарегистрируйте 2+ пользователей через `main.py` / `gui/app.py`.
2. Создайте `data/test_dataset/<Имя>/*.jpg`:
   - для уже зарегистрированных — НОВЫЕ фото (не те, что использовались при регистрации);
   - плюс папка с фото человека, которого нет в базе (impostor).
3. Запустите:
```bash
python -m evaluation.evaluate
```
Результаты — в консоли и в `evaluation/results/*.png`.

## Структура

```
config.py                  — все константы (пути, threshold, размеры)
database/db_manager.py     — SQLite: users, embeddings (много на юзера), attendance
core/face_detector.py      — MTCNN
core/face_embedder.py      — FaceNet embedding (512-D)
core/matcher.py            — cosine similarity + 1:N identification + reject option
core/camera_stream.py      — обёртка cv2.VideoCapture
core/registration.py       — регистрация нового пользователя (камера)
core/recognition.py        — identify() pipeline: камера / файл, + attendance logging
main.py                    — CLI-меню
gui/app.py                 — Tkinter GUI (тонкий слой поверх core/)
evaluation/evaluate.py     — FAR/FRR/EER/ROC/DET на отдельном test_dataset
```
