import time
import streamlit as st
from src.database.config import supabase
from src.database.db import create_attendance, enroll_student_to_subject


def show_attendance_result(df, logs):
    st.write("Please review attendance before confirming.")
    st.dataframe(df, hide_index=True, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Discard", use_container_width=True):
            st.session_state.voice_attendance_results = None
            st.session_state.attendance_images = []
            st.session_state.show_results_dialog = False  # Reset dialog state
            st.rerun()

    with col2:
        if st.button("Confirm & Save", use_container_width=True, type="primary"):
            try:
                create_attendance(logs)
                st.toast("Attendance taken successfully!")
                st.session_state.attendance_images = []
                st.session_state.voice_attendance_results = None
                st.session_state.show_results_dialog = False  # Reset dialog state
                st.rerun()
            except Exception as e:
                st.error(f"Sync failed: {e}")


@st.dialog("Attendance Reports")
def attendance_result_dialog(df, logs):
    show_attendance_result(df, logs)