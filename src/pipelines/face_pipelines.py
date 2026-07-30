import cv2
import dlib
import face_recognition_models
import numpy as np
import streamlit as st

from src.database.db import get_all_students


@st.cache_resource
def load_dlib_models():
    detector = dlib.get_frontal_face_detector()
    sp = dlib.shape_predictor(
        face_recognition_models.pose_predictor_model_location()
    )
    facerec = dlib.face_recognition_model_v1(
        face_recognition_models.face_recognition_model_location()
    )
    return detector, sp, facerec


def get_face_embeddings(image_np):
    detector, sp, facerec = load_dlib_models()

    if len(image_np.shape) == 2:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)

    # Upsample by 2 levels to catch smaller faces in classroom photos
    faces = detector(image_np, 2)

    encodings = []
    for face in faces:
        shape = sp(image_np, face)
        face_descriptor = facerec.compute_face_descriptor(image_np, shape, 1)
        encodings.append(np.array(face_descriptor, dtype=np.float64))

    return encodings


def get_trained_model():
    X = []
    y = []

    student_db = get_all_students()

    if not student_db:
        return None

    for student in student_db:
        embedding = student.get("face_embedding")
        student_id = student.get("student_id")

        if embedding is not None and student_id is not None:
            if isinstance(embedding, list):
                if len(embedding) > 0 and isinstance(embedding[0], (list, np.ndarray)):
                    for emb in embedding:
                        X.append(np.array(emb, dtype=np.float64))
                        y.append(int(student_id))
                else:
                    X.append(np.array(embedding, dtype=np.float64))
                    y.append(int(student_id))

    if len(X) == 0:
        return None

    return {"X": np.array(X), "y": np.array(y)}


def train_classifier():
    """Re-syncs cache and reloads model data from the database."""
    st.cache_resource.clear()
    model_data = get_trained_model()
    return bool(model_data)


def predict_attendance(class_image_np, allowed_student_ids=None):
    encodings = get_face_embeddings(class_image_np)
    detected_student = {}

    model_data = get_trained_model()

    if not model_data or len(encodings) == 0:
        return detected_student, [], len(encodings)

    X_train = model_data["X"]
    y_train = model_data["y"]

    if allowed_student_ids is not None and len(allowed_student_ids) > 0:
        allowed_set = set(int(sid) for sid in allowed_student_ids)
    else:
        allowed_set = set(y_train)

    resemblance_threshold = 0.72

    for encoding in encodings:
        best_match_id = None
        min_distance = float("inf")

        for idx, student_id in enumerate(y_train):
            if student_id in allowed_set:
                dist = np.linalg.norm(X_train[idx] - encoding)
                if dist < min_distance:
                    min_distance = dist
                    if dist <= resemblance_threshold:
                        best_match_id = student_id

        if best_match_id is not None:
            detected_student[best_match_id] = True

    return detected_student, list(allowed_set), len(encodings)