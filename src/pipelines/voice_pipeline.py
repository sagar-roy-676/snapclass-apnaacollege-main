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

        # Load audio from bytes at 16kHz
        audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000)
        wav = preprocess_wav(audio)
        embedding = encoder.embed_utterance(wav)

        # Ensure L2 normalization
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding.tolist()
    except Exception as e:
        st.error(f"Voice recognition error: {str(e)}")
        return None


def identify_speaker(new_embedding, candidates_dict, threshold=0.65):
    if new_embedding is None or not candidates_dict:
        return None, 0.0

    # Ensure query embedding is a 1D float numpy array and normalized
    query_emb = np.array(new_embedding, dtype=np.float32)
    query_norm = np.linalg.norm(query_emb)
    if query_norm > 0:
        query_emb = query_emb / query_norm

    best_sid = None
    best_score = -1.0

    for sid, stored_embedding in candidates_dict.items():
        if stored_embedding:
            # Convert DB vector list to numpy array
            candidate_emb = np.array(stored_embedding, dtype=np.float32)

            cand_norm = np.linalg.norm(candidate_emb)
            if cand_norm > 0:
                candidate_emb = candidate_emb / cand_norm

            # Cosine similarity calculation
            similarity = float(np.dot(query_emb, candidate_emb))

            if similarity > best_score:
                best_score = similarity
                best_sid = sid

    if best_score >= threshold:
        return best_sid, best_score

    return None, best_score


def process_bulk_audio(audio_bytes, candidates_dict, threshold=0.65):
    try:
        encoder = load_voice_encoder()

        audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000)

        # Split audio on silent pauses (top_db=30)
        segments = librosa.effects.split(audio, top_db=30)

        identified_results = {}

        for start, end in segments:
            # Ignore audio slices shorter than 0.5 seconds
            if (end - start) < sr * 0.5:
                continue

            segment_audio = audio[start:end]
            wav = preprocess_wav(segment_audio)
            embedding = encoder.embed_utterance(wav)

            sid, score = identify_speaker(
                embedding, candidates_dict, threshold
            )

            if sid:
                if (
                    sid not in identified_results
                    or score > identified_results[sid]
                ):
                    identified_results[sid] = score

        return identified_results
    except Exception as e:
        st.error(f"Bulk voice process error: {str(e)}")
        return {}