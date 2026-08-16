"""
evaluation/evaluate.py

Оценка в сценарии open-set 1:N identification с reject option.
Genuine attempt  — тестовое фото человека, который ЕСТЬ в enrollment-базе (SQLite).
Impostor attempt — тестовое фото человека, которого НЕТ в enrollment-базе.

data/test_dataset/<Имя>/*.jpg — тестовые фото, ФИЗИЧЕСКИ ОТДЕЛЬНЫЕ от фото,
использованных при регистрации (иначе — оптимистично смещённый результат,
data leakage).
"""
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

import config
from core import face_detector, face_embedder, matcher
from database import db_manager


def collect_raw_scores():
    """Возвращает список {"true_name", "predicted_name", "similarity", "is_genuine"}."""
    all_embeddings = db_manager.get_all_embeddings()
    if len(all_embeddings) == 0:
        raise RuntimeError("База пуста. Сначала зарегистрируйте пользователей.")

    if not os.path.isdir(config.TEST_DATASET_DIR):
        raise RuntimeError(f"Не найдена папка тестового датасета: {config.TEST_DATASET_DIR}")

    enrolled_names = set(name for _, name, _ in all_embeddings)
    records = []

    person_folders = [f for f in os.listdir(config.TEST_DATASET_DIR)
                       if os.path.isdir(os.path.join(config.TEST_DATASET_DIR, f))]

    for person_name in person_folders:
        folder = os.path.join(config.TEST_DATASET_DIR, person_name)
        image_files = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

        for image_file in image_files:
            frame = cv2.imread(os.path.join(folder, image_file))
            if frame is None:
                continue

            detection = face_detector.detect_face(frame)
            if detection is None:
                print(f"Лицо не найдено, пропуск: {person_name}/{image_file}")
                continue

            face = face_detector.extract_face(frame, detection)
            embedding = face_embedder.get_embedding(face)

            best_similarity, predicted_name = -1.0, None
            for _, name, stored_embedding in all_embeddings:
                sim = matcher.compute_similarity(embedding, stored_embedding)
                if sim > best_similarity:
                    best_similarity, predicted_name = sim, name

            records.append({
                "true_name": person_name,
                "predicted_name": predicted_name,
                "similarity": best_similarity,
                "is_genuine": person_name in enrolled_names,
            })

    return records


def compute_far_frr(records, threshold):
    """
    Genuine: False Reject, если similarity < threshold ИЛИ предсказано неверное имя.
    Impostor: False Accept, если similarity >= threshold.
    """
    genuine = [r for r in records if r["is_genuine"]]
    impostor = [r for r in records if not r["is_genuine"]]

    false_rejects = sum(
        1 for r in genuine
        if not (r["similarity"] >= threshold and r["predicted_name"] == r["true_name"])
    )
    false_accepts = sum(1 for r in impostor if r["similarity"] >= threshold)

    frr = false_rejects / len(genuine) if genuine else 0.0
    far = false_accepts / len(impostor) if impostor else 0.0
    return far, frr


def run_evaluation(threshold_steps=101):
    records = collect_raw_scores()

    thresholds = np.linspace(0.0, 1.0, threshold_steps)
    far_values, frr_values = [], []

    for t in thresholds:
        far, frr = compute_far_frr(records, t)
        far_values.append(far)
        frr_values.append(frr)

    far_values, frr_values = np.array(far_values), np.array(frr_values)

    eer_index = int(np.argmin(np.abs(far_values - frr_values)))
    eer_threshold = thresholds[eer_index]
    eer_value = (far_values[eer_index] + frr_values[eer_index]) / 2

    os.makedirs(config.EVAL_RESULTS_DIR, exist_ok=True)

    plt.figure()
    plt.plot(far_values, 1 - frr_values)
    plt.xlabel("FAR"); plt.ylabel("1 - FRR (TAR)"); plt.title("ROC Curve"); plt.grid(True)
    plt.savefig(os.path.join(config.EVAL_RESULTS_DIR, "roc_curve.png")); plt.close()

    plt.figure()
    plt.plot(far_values, frr_values)
    plt.xlabel("FAR"); plt.ylabel("FRR"); plt.title("DET Curve"); plt.grid(True)
    plt.savefig(os.path.join(config.EVAL_RESULTS_DIR, "det_curve.png")); plt.close()

    plt.figure()
    plt.plot(thresholds, far_values, label="FAR")
    plt.plot(thresholds, frr_values, label="FRR")
    plt.axvline(eer_threshold, color="gray", linestyle="--", label=f"EER threshold={eer_threshold:.2f}")
    plt.xlabel("Threshold"); plt.ylabel("Error Rate"); plt.title("FAR / FRR vs Threshold")
    plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(config.EVAL_RESULTS_DIR, "far_frr_vs_threshold.png")); plt.close()

    current_far, current_frr = compute_far_frr(records, config.SIMILARITY_THRESHOLD)
    genuine_count = sum(r["is_genuine"] for r in records)
    impostor_count = len(records) - genuine_count
    accuracy = sum(
        1 for r in records
        if (r["is_genuine"] and r["similarity"] >= config.SIMILARITY_THRESHOLD and r["predicted_name"] == r["true_name"])
        or (not r["is_genuine"] and r["similarity"] < config.SIMILARITY_THRESHOLD)
    ) / len(records) if records else 0.0

    print(f"\nВсего фото: {len(records)} | Genuine: {genuine_count} | Impostor: {impostor_count}")
    print(f"При config.SIMILARITY_THRESHOLD={config.SIMILARITY_THRESHOLD}: "
          f"Accuracy={accuracy:.3f}, FAR={current_far:.3f}, FRR={current_frr:.3f}")
    print(f"EER = {eer_value:.3f} при threshold = {eer_threshold:.3f}")
    print(f"Графики сохранены в {config.EVAL_RESULTS_DIR}")

    return {
        "thresholds": thresholds, "far": far_values, "frr": frr_values,
        "eer": eer_value, "eer_threshold": eer_threshold,
        "accuracy_at_config_threshold": accuracy,
    }


if __name__ == "__main__":
    run_evaluation()
