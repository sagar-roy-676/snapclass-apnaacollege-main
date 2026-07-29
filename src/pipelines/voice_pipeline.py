import io
import librosa
import numpy as np
from resemblyzer import VoiceEncoder, preprocess_wav
import streamlit as st


@st.cache_resource
def load_voice_encoder():
    return VoiceEncoder()


def get_voice_embedding(audio_bytes):
    try:
        encoder = load_voice_encoder()
        audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000)
        
        if len(audio) == 0:
            return None

        wav = preprocess_wav(audio)
        embedding = encoder.embed_utterance(wav)

        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding.tolist()
    except Exception as e:
        st.error(f"Voice recognition error: {str(e)}")
        return None


def identify_speaker(new_embedding, candidates_dict, threshold=0.45):
    """
    Compares query embedding against candidates using Cosine Similarity.
    Threshold lowered to 0.45 to account for short classroom audio clips.
    """
    if new_embedding is None or not candidates_dict:
        return None, 0.0

    query_emb = np.array(new_embedding, dtype=np.float32)
    query_norm = np.linalg.norm(query_emb)
    if query_norm > 0:
        query_emb = query_emb / query_norm

    best_sid = None
    best_score = -1.0

    # Ensure all keys are normalized as strings to avoid type mismatches
    for sid, stored_embedding in candidates_dict.items():
        if stored_embedding:
            candidate_emb = np.array(stored_embedding, dtype=np.float32)
            cand_norm = np.linalg.norm(candidate_emb)
            
            if cand_norm > 0:
                candidate_emb = candidate_emb / cand_norm

            similarity = float(np.dot(query_emb, candidate_emb))

            if similarity > best_score:
                best_score = similarity
                best_sid = str(sid)  # Always return ID as string

    if best_score >= threshold:
        return best_sid, best_score

    return None, best_score


def process_bulk_audio(audio_bytes, candidates_dict, threshold=0.45):
    try:
        encoder = load_voice_encoder()

        audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000)
        if len(audio) == 0:
            return {}

        # Split audio on silent pauses
        segments = librosa.effects.split(audio, top_db=25)

        # Fallback: if no segments split, evaluate entire audio
        if len(segments) == 0:
            segments = [(0, len(audio))]

        identified_results = {}

        for start, end in segments:
            # Slices must be at least 0.3s
            if (end - start) < sr * 0.3:
                continue

            segment_audio = audio[start:end]
            wav = preprocess_wav(segment_audio)
            
            if len(wav) == 0:
                continue

            embedding = encoder.embed_utterance(wav)

            sid, score = identify_speaker(
                embedding, candidates_dict, threshold
            )

            if sid is not None:
                sid_str = str(sid)
                if sid_str not in identified_results or score > identified_results[sid_str]:
                    identified_results[sid_str] = score

        return identified_results
    except Exception as e:
        st.error(f"Bulk voice process error: {str(e)}")
        return {}