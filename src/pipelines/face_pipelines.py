import dlib
import face_recognition_models
import numpy as np
import streamlit as st
from sklearn.svm import SVC

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
    faces = detector(image_np, 1)

    encodings = []

    for face in faces:
        shape = sp(image_np, face)
        face_descriptor = facerec.compute_face_descriptor(
            image_np, shape, 1
        )  # 128 embedding

        encodings.append(np.array(face_descriptor))
    return encodings


@st.cache_resource
def get_trained_model():
    X = []
    y = []

    student_db = get_all_students()

    if not student_db:
        return None

    for student in student_db:
        embedding = student.get("face_embedding")
        student_id = student.get("student_id")
        if embedding and student_id is not None:
            # Check if embedding is stored as a list or nested list
            if isinstance(embedding[0], (list, np.ndarray)):
                for emb in embedding:
                    X.append(np.array(emb))
                    y.append(int(student_id))
            else:
                X.append(np.array(embedding))
                y.append(int(student_id))

    if len(X) == 0:
        return None

    X = np.array(X)
    y = np.array(y)

    clf = None
    if len(set(y)) >= 2:
        clf = SVC(kernel="linear", probability=True, class_weight="balanced")
        try:
            clf.fit(X, y)
        except Exception:
            clf = None

    return {"clf": clf, "X": X, "y": y}


def train_classifier():
    st.cache_resource.clear()
    model_data = get_trained_model()
    return bool(model_data)


def predict_attendance(class_image_np, allowed_student_ids=None):
    encodings = get_face_embeddings(class_image_np)
    detected_student = {}

    model_data = get_trained_model()

    if not model_data:
        return detected_student, [], len(encodings)

    clf = model_data["clf"]
    X_train = model_data["X"]
    y_train = model_data["y"]

    all_students = sorted(list(set(y_train)))

    # Filter allowed student IDs if passed from subject selection
    if allowed_student_ids is not None:
        allowed_set = set(allowed_student_ids)
    else:
        allowed_set = set(all_students)

    resemblance_threshold = 0.50  # Lowered from 0.60 to avoid false positives between similar faces

    for encoding in encodings:
        best_match_id = None
        min_distance = float("inf")

        # Approach 1: Use classifier if available
        if clf is not None:
            try:
                probs = clf.predict_proba([encoding])[0]
                pred_idx = np.argmax(probs)
                predicted_id = int(clf.classes_[pred_idx])

                # Check confidence threshold & allowed student set
                if predicted_id in allowed_set and probs[pred_idx] > 0.45:
                    best_match_id = predicted_id
            except Exception:
                pass

        # Approach 2: Direct Minimum Distance fallback across allowed enrolled student embeddings
        if best_match_id is None:
            for idx, student_id in enumerate(y_train):
                if student_id in allowed_set:
                    dist = np.linalg.norm(X_train[idx] - encoding)
                    if dist < min_distance:
                        min_distance = dist
                        if dist <= resemblance_threshold:
                            best_match_id = student_id

        # Verification step for classifier output
        if best_match_id is not None:
            # Measure exact Euclidean distance to target student's training samples
            target_indices = np.where(y_train == best_match_id)[0]
            if len(target_indices) > 0:
                distances = [
                    np.linalg.norm(X_train[i] - encoding)
                    for i in target_indices
                ]
                if min(distances) <= resemblance_threshold:
                    detected_student[best_match_id] = True

    return detected_student, list(allowed_set), len(encodings)