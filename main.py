# DeepFake Detection System
# Streamlit app for image, audio, and video deepfake detection

import os
import io
import tempfile
import time
import math

import imageio_ffmpeg
import streamlit as st
from PIL import Image
import numpy as np
import cv2

from transformers import pipeline
import torch
import librosa
import soundfile as sf

# Make bundled ffmpeg available to librosa/soundfile
os.environ["PATH"] += os.pathsep + os.path.dirname(imageio_ffmpeg.get_ffmpeg_exe())


st.set_page_config(
    page_title="DeepFake Detection System",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="expanded",
)

DEFAULT_AUDIO_MODEL = "microsoft/wavlm-base"
DEFAULT_IMAGE_MODELS = [
    "prithivMLmods/deepfake-detector-model-v1",
    "microsoft/resnet-50",
]


@st.cache_resource(show_spinner=False)
def get_image_pipelines(model_ids):
    pipes = {}
    for model_id in model_ids:
        try:
            with st.spinner(f"Loading image model: {model_id}"):
                pipes[model_id] = pipeline(
                    "image-classification",
                    model=model_id,
                    device=0 if torch.cuda.is_available() else -1,
                )
        except Exception:
            try:
                pipes[model_id] = pipeline(
                    "image-classification",
                    model=model_id,
                    device=-1,
                )
            except Exception:
                pass
    return pipes


@st.cache_resource(show_spinner=False)
def get_audio_pipeline(model_id):
    try:
        with st.spinner(f"Loading audio model: {model_id}"):
            return pipeline(
                "audio-classification",
                model=model_id,
                device=0 if torch.cuda.is_available() else -1,
            )
    except Exception:
        try:
            return pipeline("audio-classification", model=model_id, device=-1)
        except Exception:
            return None


def parse_label_score(predictions):
    if not predictions:
        return None

    top = predictions[0]
    label = str(top.get("label", "")).lower()
    score = float(top.get("score", 0.0))

    fake_words = ["fake", "deepfake", "spoof", "manipulated", "synth", "synthetic", "generated"]
    real_words = ["real", "genuine", "bonafide", "human", "clean", "authentic", "original", "photo"]

    if any(word in label for word in fake_words):
        return score * 100.0
    if any(word in label for word in real_words):
        return (1.0 - score) * 100.0
    return score * 100.0


def extract_frames_from_video_bytes(video_bytes, max_frames=8):
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    try:
        temp_file.write(video_bytes)
        temp_file.flush()
        temp_file.close()

        cap = cv2.VideoCapture(temp_file.name)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        if total_frames <= 0:
            frames = []
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))

            if not frames:
                return [Image.new("RGB", (224, 224), color="red")]
            total_frames = len(frames)

        indices = np.linspace(0, max(0, total_frames - 1), min(max_frames, total_frames)).astype(int)
        wanted = set(indices.tolist())

        frames = []
        idx = 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx in wanted:
                frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            idx += 1

        cap.release()
        return frames if frames else [Image.new("RGB", (224, 224), color="red")]
    except Exception:
        return [Image.new("RGB", (224, 224), color="red")]
    finally:
        try:
            os.unlink(temp_file.name)
        except Exception:
            pass


def run_image_pipeline(image_bytes, image_pipes):
    try:
        pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        per_model_scores = []
        details = {}

        for model_id, pipe in image_pipes.items():
            try:
                preds = pipe(pil)
                score = parse_label_score(preds)
                if score is None:
                    score = float(preds[0].get("score", 0.0)) * 100.0
                per_model_scores.append(score)
                details[model_id] = preds
            except Exception as exc:
                details[model_id] = {"error": str(exc)}

        final_score = float(np.mean(per_model_scores)) if per_model_scores else None
        return final_score, per_model_scores, details, pil
    except Exception as exc:
        return None, [], {"error": str(exc)}, None


def run_audio_pipeline(audio_bytes, audio_pipe):
    if audio_pipe is None:
        return None, {"error": "Audio model not loaded"}

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    try:
        try:
            y, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000, mono=True)
            sf.write(temp_file.name, y, sr, subtype="PCM_16")
        except Exception as exc:
            return None, {"error": f"Failed to process audio: {exc}"}

        preds = audio_pipe(temp_file.name)
        score = parse_label_score(preds)
        if score is None:
            score = float(preds[0].get("score", 0.0)) * 100.0

        return score, preds
    except Exception as exc:
        return None, {"error": f"Failed to process audio: {exc}"}
    finally:
        try:
            os.unlink(temp_file.name)
        except Exception:
            pass


def run_video_pipeline(video_bytes, image_pipes, max_frames=8):
    frames = extract_frames_from_video_bytes(video_bytes, max_frames=max_frames)
    if not frames:
        return None, {"error": "No frames extracted"}

    frame_scores = []
    frame_details = []

    for pil in frames:
        per_model_scores = []
        details = {}

        for model_id, pipe in image_pipes.items():
            try:
                preds = pipe(pil)
                score = parse_label_score(preds)
                if score is None:
                    score = float(preds[0].get("score", 0.0)) * 100.0
                per_model_scores.append(score)
                details[model_id] = preds
            except Exception as exc:
                details[model_id] = {"error": str(exc)}

        frame_scores.append(float(np.mean(per_model_scores)) if per_model_scores else None)
        frame_details.append(details)

    valid_scores = [s for s in frame_scores if s is not None]
    mean_score = float(np.mean(valid_scores)) if valid_scores else None
    top_k = max(1, int(math.ceil(0.2 * len(valid_scores)))) if valid_scores else 1
    topk_mean = float(np.mean(sorted(valid_scores, reverse=True)[:top_k])) if valid_scores else None
    final_score = max(mean_score or 0.0, topk_mean or 0.0)

    return final_score, {
        "frame_scores": frame_scores,
        "frame_details": frame_details,
        "mean": mean_score,
        "topk_mean": topk_mean,
    }


st.title("🔍 DeepFake Detection System")
st.caption("Detect AI-generated or manipulated images, audio, and video.")

with st.sidebar:
    st.header("Settings")

    with st.expander("Advanced Settings"):
        image_model_ids = st.text_area("Image Model IDs", value=", ".join(DEFAULT_IMAGE_MODELS))
        audio_model_id = st.text_input("Audio Model ID", value=DEFAULT_AUDIO_MODEL)
        max_frames = st.slider("Video Frames to Analyze", 1, 16, 8)

    threshold = st.slider(
        "Confidence Threshold",
        0,
        100,
        60,
        help="Higher values = stricter detection",
    )
    show_details = st.checkbox("Show Technical Details", value=False)

    st.divider()
    st.subheader("About")
    st.write(
        "DeepFake Detection System uses ML models to detect "
        "AI-generated or manipulated media."
    )
    st.write("All processing happens locally on your system.")
    st.divider()
    st.write("Supported formats:")
    st.write("Images: JPG, PNG, WebP")
    st.write("Audio: WAV, MP3, M4A")
    st.write("Video: MP4, MOV")

image_model_list = [m.strip() for m in image_model_ids.split(",") if m.strip()]
img_pipes = get_image_pipelines(image_model_list)
audio_pipe = get_audio_pipeline(audio_model_id)

st.subheader("Upload Content")
upload = st.file_uploader(
    "Choose an image, audio, or video file",
    type=["png", "jpg", "jpeg", "webp", "mp4", "mov", "wav", "mp3", "m4a"],
)

analyze_btn = st.button("Analyze", type="primary", use_container_width=True)


def show_verdict(score, label="AI Likelihood Score"):
    st.metric(label, f"{score:.1f}%")
    st.progress(score / 100)

    is_fake = score >= threshold
    verdict = "FAKE" if is_fake else "REAL"
    verdict_color = "#dc2626" if is_fake else "#16a34a"

    st.markdown(
        f"""
        <div style="text-align:center; margin-top:0.8rem;">
            <div style="font-size:3rem; font-weight:800; color:{verdict_color}; letter-spacing:1px;">
                {verdict}
            </div>
            <div style="font-size:1rem; color:#6b7280;">Result</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


if analyze_btn and upload:
    start = time.time()
    st.divider()
    st.subheader("Results")

    data = upload.read()
    file_name = (upload.name or "").lower()
    mime = (upload.type or "").lower()

    is_image = any(file_name.endswith(ext) for ext in (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp")) or mime.startswith("image")
    is_audio = any(file_name.endswith(ext) for ext in (".wav", ".mp3", ".m4a", ".flac", ".ogg")) or mime.startswith("audio")
    is_video = any(file_name.endswith(ext) for ext in (".mp4", ".mov", ".avi", ".mkv", ".webm")) or mime.startswith("video")

    if is_image:
        with st.spinner("Analyzing image..."):
            score, model_scores, details, pil = run_image_pipeline(data, img_pipes)

        if pil:
            st.image(pil, use_container_width=True)

        if score is None:
            st.warning("No valid model responses.")
        else:
            show_verdict(score)

        if show_details:
            with st.expander("Technical Details"):
                st.json({"per_model_scores": model_scores, "model_outputs": details})

    elif is_audio:
        st.audio(data)
        with st.spinner("Analyzing audio..."):
            score, details = run_audio_pipeline(data, audio_pipe)

        if score is None:
            st.warning(details.get("error", "Audio model failed to return a score."))
        else:
            show_verdict(score)

        if show_details:
            with st.expander("Technical Details"):
                st.json(details)

    elif is_video:
        st.video(data)
        with st.spinner("Analyzing video frames..."):
            score, details = run_video_pipeline(data, img_pipes, max_frames=max_frames)

        if score is None:
            st.warning("No valid frame scores.")
        else:
            show_verdict(score)

        if show_details:
            with st.expander("Technical Details"):
                st.json(details)

    else:
        st.warning("Unsupported file type. Please upload an image, audio, or video file.")

    elapsed = time.time() - start
    st.caption(f"Completed in {elapsed:.2f}s")


st.divider()
st.caption("Built with Streamlit | Python | Machine Learning")
st.caption("This project aims to raise awareness and promote responsible use of AI technologies.")
