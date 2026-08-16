"""
matcher.py
Cosine similarity между embeddings + 1:N identification с reject option.
"""

from sklearn.metrics.pairwise import cosine_similarity

import config
from database import db_manager


def compute_similarity(embedding_a, embedding_b):
    """Возвращает cosine similarity (float, диапазон [-1, 1])."""
    a = embedding_a.reshape(1, -1)
    b = embedding_b.reshape(1, -1)
    return cosine_similarity(a, b)[0][0]


def find_match(new_embedding):
    """
    Сравнивает new_embedding со ВСЕМИ embeddings в базе, берёт максимум.
    Если максимум < threshold — возвращает Unknown (reject option).

    Возвращает (user_id, name, similarity). user_id=None, если Unknown.
    """
    all_embeddings = db_manager.get_all_embeddings()

    if len(all_embeddings) == 0:
        return None, "Unknown", 0.0

    best_user_id, best_name, best_similarity = None, "Unknown", -1.0

    for user_id, name, stored_embedding in all_embeddings:
        similarity = compute_similarity(new_embedding, stored_embedding)
        if similarity > best_similarity:
            best_similarity = similarity
            best_name = name
            best_user_id = user_id

    if best_similarity < config.SIMILARITY_THRESHOLD:
        return None, "Unknown", best_similarity

    return best_user_id, best_name, best_similarity
