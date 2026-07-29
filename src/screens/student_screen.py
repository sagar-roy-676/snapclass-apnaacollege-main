import time
import numpy as np
from PIL import Image
import streamlit as st

from src.database.db import (
    create_student,
    get_all_students,
    get_student_attendance,
    get_student_subjects,
    unenroll_student_to_subject,
)
from src.pipelines.face_pipelines import (
    get_face_embeddings,
    predict_attendance,
    train_classifier,
)
from src.pipelines.voice_pipeline import get_voice_embedding
from src.screens.components.dialog_enroll import enroll_dialog
from src.screens.components.footer import footer_dashboard
from src.screens.components.header import header_dashboard
from src.screens.components.subject_card import subject_card
from src.ui.base_layout import style_background_dashboard, style_base_layout


def student_dashboard():
    student_data = st.session_state.student_data
    student_id = student_data["student_id"]

    c1, c2 = st.columns(2, vertical_alignment="center", gap="large")
    with c1:
        header_dashboard()
    with c2:
        st.subheader(f"Welcome, {student_data['name']}")
        if st.button(
            "Logout",
            type="secondary",
            key="student_dashboard_logout_btn",
            shortcut="control+backspace",
        ):
            st.session_state["is_logged_in"] = False
            if "student_data" in st.session_state:
                del st.session_state["student_data"]
            st.rerun()

    st.write("")  # Replaced st.space() with standard spacing

    c1, c2 = st.columns(2)
    with c1:
        st.header("Your Enrolled Subjects")
    with c2:
        if st.button(
            "Enroll in Subject",
            type="primary",
            use_container_width=True,
            key="enroll_in_subject_dialog_btn",
        ):
            enroll_dialog()

    st.divider()

    with st.spinner("Loading your enrolled subjects..."):
        subjects = get_student_subjects(student_id)
        logs = get_student_attendance(student_id)

    stats_map = {}
    if logs:
        for log in logs:
            sid = log["subject_id"]
            if sid not in stats_map:
                stats_map[sid] = {"total": 0, "attended": 0}

            stats_map[sid]["total"] += 1
            if log.get("is_present"):
                stats_map[sid]["attended"] += 1

    if subjects:
        cols = st.columns(2)
        for i, sub_node in enumerate(subjects):
            sub = sub_node["subjects"]
            sid = sub["subject_id"]
            stats = stats_map.get(sid, {"total": 0, "attended": 0})

            def make_unenroll_callback(subject_obj, subject_id_val, index):
                def unenroll_button():
                    if st.button(
                        "Unenroll from this course",
                        type="tertiary",
                        use_container_width=True,
                        icon=":material/delete_forever:",
                        key=f"unenroll_btn_{subject_id_val}_{index}",
                    ):
                        unenroll_student_to_subject(student_id, subject_id_val)
                        st.toast(
                            f"Unenrolled from {subject_obj['name']} successfully!"
                        )
                        time.sleep(1)
                        st.rerun()

                return unenroll_button

            with cols[i % 2]:
                subject_card(
                    name=sub["name"],
                    code=sub["subject_code"],
                    section=sub.get("section", ""),
                    stats=[
                        ("📅", "Total", stats["total"]),
                        ("✅", "Attended", stats["attended"]),
                    ],
                    footer_callback=make_unenroll_callback(sub, sid, i),
                )
    else:
        st.info("You are not enrolled in any subjects yet.")

    footer_dashboard()


def student_screen():
    style_background_dashboard()
    style_base_layout()

    if "student_data" in st.session_state:
        student_dashboard()
        return

    c1, c2 = st.columns(2, vertical_alignment="center", gap="large")
    with c1:
        header_dashboard()
    with c2:
        if st.button(
            "Go back to Home",
            type="secondary",
            key="faceid_login_back_home_btn",
            shortcut="control+backspace",
        ):
            st.session_state["login_type"] = None
            st.rerun()

    st.header("Login using FaceID")
    st.write("")

    photo_source = st.camera_input(
        "Position your face in the center", key="student_camera_login_input"
    )

    if photo_source:
        img = np.array(Image.open(photo_source))

        with st.spinner("AI is scanning..."):
            detected, all_ids, num_faces = predict_attendance(img)

            if num_faces == 0:
                st.warning("Face not found!")
            elif num_faces > 1:
                st.warning("Multiple faces found!")
            else:
                if detected:
                    student_id = list(detected.keys())[0]
                    all_students = get_all_students()
                    student = next(
                        (
                            s
                            for s in all_students
                            if s["student_id"] == student_id
                        ),
                        None,
                    )

                    if student:
                        st.session_state.is_logged_in = True
                        st.session_state.user_role = "student"
                        st.session_state.student_data = student
                        st.toast(f"Welcome Back {student['name']}")
                        time.sleep(1)
                        st.rerun()
                else:
                    st.info("Face not recognized! You might be a new student.")

                    # Use st.form to preserve form inputs across user interactions/reruns
                    with st.form("new_student_registration_form"):
                        st.header("Register new Profile")
                        new_name = st.text_input(
                            "Enter your name",
                            placeholder="E.g. Sagar Roy",
                            key="reg_student_name_input",
                        )

                        st.subheader("Optional : Voice Enrollment")
                        st.info("Enroll your voice for voice-only attendance")

                        audio_data = st.audio_input(
                            "Record a short phrase like 'I am present, My name is Akash.'",
                            key="voice_enrollment_audio",
                        )

                        submit_reg = st.form_submit_button(
                            "Create Account", type="primary"
                        )

                        if submit_reg:
                            if new_name:
                                with st.spinner("Creating profile..."):
                                    encodings = get_face_embeddings(img)
                                    if encodings:
                                        face_emb = encodings[0].tolist()

                                        voice_emb = None
                                        if audio_data:
                                            voice_bytes = audio_data.getvalue()
                                            voice_emb = get_voice_embedding(
                                                voice_bytes
                                            )

                                        response_data = create_student(
                                            new_name,
                                            face_embedding=face_emb,
                                            voice_embedding=voice_emb,
                                        )

                                        if response_data:
                                            train_classifier()
                                            st.session_state.is_logged_in = True
                                            st.session_state.user_role = (
                                                "student"
                                            )

                                            # Safe list or object parsing
                                            created_data = (
                                                response_data[0]
                                                if isinstance(
                                                    response_data, list
                                                )
                                                else response_data
                                            )
                                            st.session_state.student_data = (
                                                created_data
                                            )

                                            st.toast(
                                                f"Profile Created! Hi {new_name}!"
                                            )
                                            time.sleep(1)
                                            st.rerun()
                                    else:
                                        st.error(
                                            "Could not capture facial features for registration."
                                        )
                            else:
                                st.warning("Please enter your name!")

    footer_dashboard()