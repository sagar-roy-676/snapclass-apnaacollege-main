from datetime import datetime
from zoneinfo import ZoneInfo
import pandas as pd
import streamlit as st

from src.database.config import supabase
from src.pipelines.voice_pipeline import process_bulk_audio

IST_TZ = ZoneInfo("Asia/Kolkata")


@st.dialog("Voice Attendance")
def voice_attendance_dialog(selected_subject_id):
    st.write(
        "Record audio of students saying 'I am present'. The AI will recognize their voices."
    )

    audio_data = st.audio_input("Record classroom audio")

    st.caption(
        "⚠️ *Ensure browser mic permission is granted and speak clearly into the microphone.*"
    )

    if st.button("Analyze Audio", use_container_width=True, type="primary"):
        if audio_data is None:
            st.warning("Please record classroom audio first before analyzing!")
            return

        with st.spinner("Processing Audio data..."):
            enrolled_res = (
                supabase.table("subject_students")
                .select("*, students(*)")
                .eq("subject_id", selected_subject_id)
                .execute()
            )
            enrolled_students = enrolled_res.data if enrolled_res else []

            if not enrolled_students:
                st.warning("No students enrolled in this course.")
                return

            # Force key to string format: str(student_id)
            candidates_dict = {
                str(s["students"]["student_id"]): s["students"]["voice_embedding"]
                for s in enrolled_students
                if s["students"] and s["students"].get("voice_embedding")
            }

            if not candidates_dict:
                st.error("No enrolled students have voice profiles registered.")
                return

            audio_bytes = audio_data.getvalue()

            # detected_scores keys are strings
            detected_scores = process_bulk_audio(audio_bytes, candidates_dict, threshold=0.45)

            results, attendance_to_log = [], []

            current_timestamp = datetime.now(IST_TZ).strftime(
                "%Y-%m-%dT%H:%M:%S"
            )

            for node in enrolled_students:
                student = node["students"]
                sid_str = str(student["student_id"])
                
                # Safe lookup using string ID
                score = detected_scores.get(sid_str, 0.0)
                is_present = bool(score > 0)

                score_str = f"{score * 100:.1f}% Match" if is_present else "-"

                results.append(
                    {
                        "Name": student["name"],
                        "ID": student["student_id"],
                        "Source": score_str,
                        "Status": "✅ Present" if is_present else "❌ Absent",
                    }
                )

                attendance_to_log.append(
                    {
                        "student_id": student["student_id"],
                        "subject_id": selected_subject_id,
                        "timestamp": current_timestamp,
                        "is_present": bool(is_present),
                    }
                )

            st.session_state.voice_attendance_results = (
                pd.DataFrame(results),
                attendance_to_log,
            )
            st.session_state.show_results_dialog = True

            st.rerun()