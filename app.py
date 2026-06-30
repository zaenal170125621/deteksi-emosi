from __future__ import annotations

import os
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, cast

# Kurangi log TensorFlow yang terlalu ramai di terminal Streamlit.
_ = os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import av
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from deepface import DeepFace
from numpy.typing import NDArray
from streamlit.delta_generator import DeltaGenerator
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer
from ultralytics import YOLO
from yt_dlp import YoutubeDL

DEFAULT_YOLO_WEIGHTS = "yolo26n.pt"
Frame = NDArray[np.uint8]
RTC_CONFIGURATION = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}

EMOTION_ID = {
    "angry": "marah",
    "disgust": "jijik",
    "fear": "takut",
    "happy": "senang",
    "sad": "sedih",
    "surprise": "terkejut",
    "neutral": "netral",
}

# Warna dalam format BGR karena frame diproses oleh OpenCV.
EMOTION_COLORS = {
    "angry": (36, 28, 237),
    "disgust": (23, 138, 0),
    "fear": (168, 70, 168),
    "happy": (0, 196, 255),
    "sad": (204, 113, 54),
    "surprise": (255, 144, 30),
    "neutral": (170, 170, 170),
}

VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
DEEPFACE_BACKENDS = ["opencv", "retinaface", "ssd", "mtcnn"]


@dataclass(frozen=True)
class EmotionDetection:
    bbox: tuple[int, int, int, int]
    emotion: str
    score: float
    source: str


@st.cache_resource(show_spinner="Memuat model YOLO...")
def load_yolo(weights: str) -> YOLO:
    return YOLO(weights)


@st.cache_resource(show_spinner="Memuat model emosi DeepFace...")
def warmup_deepface() -> bool:
    DeepFace.build_model("Emotion", task="facial_attribute")
    return True


def clamp_box(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    width: int,
    height: int,
) -> tuple[int, int, int, int] | None:
    x1 = max(0, min(width - 1, int(x1)))
    y1 = max(0, min(height - 1, int(y1)))
    x2 = max(0, min(width, int(x2)))
    y2 = max(0, min(height, int(y2)))

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


def normalize_deepface_output(output: object) -> list[dict[str, Any]]:
    if isinstance(output, list):
        return [item for item in output if isinstance(item, dict)]
    if isinstance(output, dict):
        return [output]
    return []


def emotion_label(emotion: str, score: float) -> str:
    translated = EMOTION_ID.get(emotion, emotion)
    return f"{translated} ({score:.1f}%)"


def draw_label(
    frame: Frame, text: str, x: int, y: int, color: tuple[int, int, int]
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 2
    padding = 6
    (text_width, text_height), baseline = cv2.getTextSize(
        text, font, font_scale, thickness
    )

    top = max(0, y - text_height - padding * 2)
    right = min(frame.shape[1], x + text_width + padding * 2)
    bottom = min(frame.shape[0], top + text_height + padding * 2 + baseline)

    cv2.rectangle(frame, (x, top), (right, bottom), color, -1)
    cv2.putText(
        frame,
        text,
        (x + padding, top + text_height + padding // 2),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def draw_emotion(frame: Frame, detection: EmotionDetection) -> None:
    x1, y1, x2, y2 = detection.bbox
    color = EMOTION_COLORS.get(detection.emotion, (90, 190, 255))
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    draw_label(frame, emotion_label(detection.emotion, detection.score), x1, y1, color)


def deepface_emotion_from_crop(
    crop_bgr: Frame,
    origin: tuple[int, int],
    frame_shape: tuple[int, ...],
    detector_backend: str,
    direct_face_crop: bool,
    store_error: bool = True,
) -> list[EmotionDetection]:
    if crop_bgr.size == 0:
        return []

    crop_h, crop_w = crop_bgr.shape[:2]
    if crop_w < 20 or crop_h < 20:
        return []

    backend = "skip" if direct_face_crop else detector_backend

    try:
        output = DeepFace.analyze(
            img_path=crop_bgr,
            actions=["emotion"],
            detector_backend=backend,
            enforce_detection=False,
            silent=True,
        )
    except Exception as exc:
        # DeepFace kadang gagal pada crop buram/terpotong. Lewati frame/crop itu
        # agar stream tetap berjalan.
        if store_error:
            st.session_state["last_deepface_error"] = str(exc)
        return []

    frame_h, frame_w = frame_shape[:2]
    origin_x, origin_y = origin
    detections: list[EmotionDetection] = []

    for item in normalize_deepface_output(output):
        emotion = str(item.get("dominant_emotion", "unknown"))
        scores = item.get("emotion", {})
        score = float(scores.get(emotion, 0.0)) if isinstance(scores, dict) else 0.0

        if direct_face_crop:
            box = clamp_box(
                origin_x,
                origin_y,
                origin_x + crop_w,
                origin_y + crop_h,
                frame_w,
                frame_h,
            )
        else:
            region = (
                item.get("region", {})
                if isinstance(item.get("region", {}), dict)
                else {}
            )
            rx = int(region.get("x", 0) or 0)
            ry = int(region.get("y", 0) or 0)
            rw = int(region.get("w", crop_w) or crop_w)
            rh = int(region.get("h", crop_h) or crop_h)
            box = clamp_box(
                origin_x + rx,
                origin_y + ry,
                origin_x + rx + rw,
                origin_y + ry + rh,
                frame_w,
                frame_h,
            )

        if box is None:
            continue

        detections.append(
            EmotionDetection(
                bbox=box,
                emotion=emotion,
                score=score,
                source="yolo-face" if direct_face_crop else "deepface-face",
            )
        )

    return detections


def should_treat_yolo_as_face_model(mode: str, class_names: dict[int, str]) -> bool:
    if mode == "Face model":
        return True
    if mode == "COCO/person":
        return False

    names = [str(name).lower() for name in class_names.values()]
    return any("face" in name or "wajah" in name for name in names)


def yolo_boxes(
    frame_bgr: Frame,
    model: YOLO,
    confidence: float,
    yolo_mode: str,
) -> tuple[list[tuple[int, int, int, int, str, float]], bool]:
    result = model.predict(frame_bgr, conf=confidence, verbose=False)[0]
    class_names = {int(k): str(v) for k, v in result.names.items()}
    use_face_boxes = should_treat_yolo_as_face_model(yolo_mode, class_names)
    frame_h, frame_w = frame_bgr.shape[:2]
    boxes: list[tuple[int, int, int, int, str, float]] = []
    if result.boxes is None:
        return boxes, use_face_boxes

    for box in result.boxes:
        class_id = int(box.cls.item())
        class_name = class_names.get(class_id, str(class_id))

        # Kalau memakai YOLO COCO bawaan, gunakan class person saja. Wajah dicari
        # oleh DeepFace di dalam crop person.
        if not use_face_boxes and class_name.lower() != "person":
            continue

        xyxy = box.xyxy[0].tolist()
        clamped = clamp_box(xyxy[0], xyxy[1], xyxy[2], xyxy[3], frame_w, frame_h)
        if clamped is None:
            continue

        boxes.append((*clamped, class_name, float(box.conf.item())))

    return boxes, use_face_boxes


def process_frame(
    frame_bgr: Frame,
    model: YOLO,
    confidence: float,
    yolo_mode: str,
    deepface_backend: str,
    store_deepface_error: bool = True,
) -> tuple[Frame, list[EmotionDetection]]:
    annotated = frame_bgr.copy()
    detections: list[EmotionDetection] = []
    boxes, use_face_boxes = yolo_boxes(frame_bgr, model, confidence, yolo_mode)

    for x1, y1, x2, y2, class_name, box_conf in boxes:
        crop = frame_bgr[y1:y2, x1:x2]

        if use_face_boxes:
            detections.extend(
                deepface_emotion_from_crop(
                    crop,
                    origin=(x1, y1),
                    frame_shape=frame_bgr.shape,
                    detector_backend=deepface_backend,
                    direct_face_crop=True,
                    store_error=store_deepface_error,
                )
            )
        else:
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (145, 145, 145), 1)
            draw_label(annotated, f"{class_name} {box_conf:.2f}", x1, y1, (90, 90, 90))
            detections.extend(
                deepface_emotion_from_crop(
                    crop,
                    origin=(x1, y1),
                    frame_shape=frame_bgr.shape,
                    detector_backend=deepface_backend,
                    direct_face_crop=False,
                    store_error=store_deepface_error,
                )
            )

    for detection in detections:
        draw_emotion(annotated, detection)

    return annotated, detections


class RealtimeEmotionProcessor(VideoProcessorBase):
    def __init__(
        self,
        model: YOLO,
        confidence: float,
        yolo_mode: str,
        deepface_backend: str,
        frame_step: int,
    ) -> None:
        self.model = model
        self.confidence = confidence
        self.yolo_mode = yolo_mode
        self.deepface_backend = deepface_backend
        self.frame_step = frame_step
        self.frame_index = 0
        self.last_detections: list[EmotionDetection] = []
        self.counts: Counter[str] = Counter()
        self.lock = Lock()

    def update_settings(
        self,
        confidence: float,
        yolo_mode: str,
        deepface_backend: str,
        frame_step: int,
    ) -> None:
        with self.lock:
            self.confidence = confidence
            self.yolo_mode = yolo_mode
            self.deepface_backend = deepface_backend
            self.frame_step = frame_step

    def get_counts(self) -> Counter[str]:
        with self.lock:
            return Counter(self.counts)

    def reset_counts(self) -> None:
        with self.lock:
            self.counts.clear()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image_rgb = frame.to_ndarray(format="rgb24")
        frame_bgr = cast(Frame, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))

        with self.lock:
            confidence = self.confidence
            yolo_mode = self.yolo_mode
            deepface_backend = self.deepface_backend
            frame_step = max(1, self.frame_step)
            frame_index = self.frame_index
            self.frame_index += 1

        try:
            if frame_index % frame_step == 0:
                annotated, detections = process_frame(
                    frame_bgr,
                    model=self.model,
                    confidence=confidence,
                    yolo_mode=yolo_mode,
                    deepface_backend=deepface_backend,
                    store_deepface_error=False,
                )
                with self.lock:
                    self.last_detections = detections
                    self.counts.update(detection.emotion for detection in detections)
            else:
                annotated = frame_bgr.copy()
                with self.lock:
                    detections = list(self.last_detections)
                for detection in detections:
                    draw_emotion(annotated, detection)
        except Exception:
            annotated = frame_bgr

        output_rgb = cast(Frame, cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
        return av.VideoFrame.from_ndarray(output_rgb, format="rgb24")


def counter_to_dataframe(counter: Counter[str]) -> pd.DataFrame:
    rows = [
        {"emosi": EMOTION_ID.get(emotion, emotion), "jumlah": count}
        for emotion, count in counter.most_common()
    ]
    return pd.DataFrame(rows)


def render_summary(counter: Counter[str], container: DeltaGenerator) -> None:
    with container.container():
        if not counter:
            st.info("Belum ada orang/emosi yang terdeteksi.")
            return

        df = counter_to_dataframe(counter)
        st.dataframe(df, width="stretch", hide_index=True)
        st.bar_chart(df.set_index("emosi"))


def save_uploaded_file(uploaded_file, suffix: str) -> Path:
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        _ = tmp.write(uploaded_file.getbuffer())
        return Path(tmp.name)


def download_youtube_video(url: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    options: dict[str, Any] = {
        # Ambil satu file video langsung agar tidak perlu ffmpeg untuk merge audio/video.
        # Audio tidak dibutuhkan karena aplikasi hanya menganalisis frame gambar.
        "format": "best[ext=mp4][vcodec!=none][height<=720]/best[vcodec!=none][height<=720]/best",
        "outtmpl": str(output_dir / "%(title).80s.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
    }

    with YoutubeDL(cast(Any, options)) as ydl:
        _ = ydl.extract_info(url, download=True)

    candidates = sorted(
        [
            path
            for path in output_dir.iterdir()
            if path.suffix.lower() in VIDEO_SUFFIXES
        ],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            "Video hasil download tidak ditemukan atau formatnya tidak didukung OpenCV."
        )

    return candidates[0]


def process_video(
    video_path: Path,
    model: YOLO,
    confidence: float,
    yolo_mode: str,
    deepface_backend: str,
    frame_step: int,
    max_frames: int,
) -> Counter[str]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Tidak bisa membuka video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    target_frames = max_frames
    if total_frames > 0:
        target_frames = min(max_frames, max(1, total_frames // max(1, frame_step)))

    progress = st.progress(0)
    status = st.empty()
    preview = st.empty()
    summary = st.empty()
    counts: Counter[str] = Counter()

    processed = 0
    frame_index = 0

    try:
        while processed < max_frames:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            if frame_index % frame_step != 0:
                frame_index += 1
                continue

            frame_bgr = cast(Frame, frame_bgr)
            annotated, detections = process_frame(
                frame_bgr,
                model=model,
                confidence=confidence,
                yolo_mode=yolo_mode,
                deepface_backend=deepface_backend,
            )
            counts.update(detection.emotion for detection in detections)

            processed += 1
            frame_index += 1
            progress.progress(min(processed / max(1, target_frames), 1.0))
            status.write(
                f"Memproses frame ke-{frame_index} | frame dianalisis: {processed}"
            )
            preview.image(
                cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                caption="Preview frame terakhir",
                channels="RGB",
                width="stretch",
            )
            render_summary(counts, summary)
    finally:
        cap.release()

    progress.progress(1.0)
    status.success(f"Selesai. Total frame dianalisis: {processed}")
    return counts


def load_models_from_sidebar() -> tuple[YOLO, str, float, str, int, int]:
    with st.sidebar:
        st.header("Pengaturan model")
        yolo_weights = st.text_input(
            "YOLO weights / model",
            value=DEFAULT_YOLO_WEIGHTS,
            help=(
                "Default memakai YOLO26 COCO untuk mendeteksi person. "
                "Untuk deteksi wajah langsung, isi path model face seperti models/yolo26n-face.pt."
            ),
        )
        yolo_mode = st.radio(
            "Tipe output YOLO",
            options=["COCO/person", "Face model", "Otomatis"],
            index=0,
            help=(
                "COCO/person: YOLO mencari person lalu DeepFace mencari wajah di crop person. "
                "Face model: YOLO dianggap langsung mendeteksi wajah."
            ),
        )
        confidence = st.slider("Confidence YOLO", 0.10, 0.90, 0.35, 0.05)
        deepface_backend = st.radio(
            "Backend deteksi wajah DeepFace", DEEPFACE_BACKENDS, index=0
        )

        st.header("Pengaturan video")
        frame_step = st.slider(
            "Analisis tiap N frame",
            min_value=1,
            max_value=60,
            value=15,
            help="Naikkan nilai ini agar video lebih cepat diproses.",
        )
        max_frames = st.slider("Maksimal frame dianalisis", 10, 1000, 150, 10)

    _ = warmup_deepface()
    model = load_yolo(yolo_weights.strip())
    return model, yolo_mode, confidence, deepface_backend, frame_step, max_frames


def show_last_deepface_error() -> None:
    error = st.session_state.get("last_deepface_error")
    if error:
        with st.expander("Detail error DeepFace terakhir"):
            st.code(error)


def main() -> None:
    st.set_page_config(page_title="Deteksi Emosi DeepFace + YOLO26", layout="wide")
    st.title("Deteksi Emosi dengan DeepFace, YOLO26, Streamlit, dan yt-dlp")
    st.write(
        "Aplikasi ini bisa menganalisis gambar, video lokal, atau link YouTube. "
        + "YOLO26 dipakai untuk menemukan target, lalu DeepFace memprediksi emosi wajah."
    )

    try:
        model, yolo_mode, confidence, deepface_backend, frame_step, max_frames = (
            load_models_from_sidebar()
        )
    except Exception as exc:
        st.error(
            "Model gagal dimuat. Periksa nama/path YOLO weights dan instalasi dependensi."
        )
        st.exception(exc)
        return

    source = st.tabs(["Gambar", "Video", "YouTube", "Realtime"])

    with source[0]:
        image_file = st.file_uploader(
            "Upload gambar", type=["jpg", "jpeg", "png", "webp"]
        )
        if image_file is not None:
            file_bytes = np.frombuffer(image_file.getvalue(), np.uint8)
            frame_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if frame_bgr is None:
                st.error("Gambar tidak bisa dibaca.")
            else:
                frame_bgr = cast(Frame, frame_bgr)
                with st.spinner("Menganalisis gambar..."):
                    annotated, detections = process_frame(
                        frame_bgr,
                        model=model,
                        confidence=confidence,
                        yolo_mode=yolo_mode,
                        deepface_backend=deepface_backend,
                    )
                st.image(
                    cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                    channels="RGB",
                    width="stretch",
                )
                render_summary(
                    Counter(detection.emotion for detection in detections), st.empty()
                )
                show_last_deepface_error()

    with source[1]:
        video_file = st.file_uploader(
            "Upload video", type=["mp4", "mov", "avi", "mkv", "webm"]
        )
        if video_file is not None and st.button("Proses video", type="primary"):
            suffix = Path(video_file.name).suffix or ".mp4"
            video_path = save_uploaded_file(video_file, suffix)
            try:
                with st.spinner("Memproses video..."):
                    process_video(
                        video_path,
                        model=model,
                        confidence=confidence,
                        yolo_mode=yolo_mode,
                        deepface_backend=deepface_backend,
                        frame_step=frame_step,
                        max_frames=max_frames,
                    )
                show_last_deepface_error()
            finally:
                video_path.unlink(missing_ok=True)

    with source[2]:
        st.info(
            "Gunakan hanya video yang Anda punya hak untuk unduh/proses. "
            + "yt-dlp membutuhkan koneksi internet. Audio tidak diunduh karena hanya frame video yang dianalisis."
        )
        youtube_url = st.text_input("Link YouTube")
        if st.button("Download dan proses YouTube", type="primary"):
            if not youtube_url.strip():
                st.warning("Masukkan link YouTube terlebih dahulu.")
            else:
                with tempfile.TemporaryDirectory(
                    prefix="deteksi_emosi_yt_"
                ) as temp_dir:
                    try:
                        with st.spinner("Mengunduh video dari YouTube..."):
                            video_path = download_youtube_video(
                                youtube_url.strip(), Path(temp_dir)
                            )
                        st.success(f"Video berhasil diunduh: {video_path.name}")
                        with st.spinner("Memproses video..."):
                            process_video(
                                video_path,
                                model=model,
                                confidence=confidence,
                                yolo_mode=yolo_mode,
                                deepface_backend=deepface_backend,
                                frame_step=frame_step,
                                max_frames=max_frames,
                            )
                        show_last_deepface_error()
                    except Exception as exc:
                        st.error("Gagal mengunduh atau memproses video YouTube.")
                        st.exception(exc)

    with source[3]:
        st.warning(
            "Kamera realtime di HP hanya bisa berjalan dari halaman aman: localhost atau HTTPS. "
            "Jika dibuka dari http://192.168.x.x:8501, browser akan menolak kamera dan bisa menampilkan error komponen."
        )
        with st.expander("Cara buka realtime dari HP"):
            st.markdown(
                "1. Jalankan app di laptop seperti biasa.\n"
                "2. Buat URL HTTPS dengan tunnel, misalnya `ngrok http 8501`.\n"
                "3. Buka URL `https://...ngrok-free.app` dari HP.\n"
                "4. Pilih kamera, klik `START`, lalu izinkan akses kamera."
            )

        camera_choice = st.radio(
            "Kamera",
            options=["Depan", "Belakang"],
            horizontal=True,
            key="realtime_camera_choice",
        )
        facing_mode = "user" if camera_choice == "Depan" else "environment"
        realtime_frame_step = st.slider(
            "Analisis realtime tiap N frame",
            min_value=1,
            max_value=30,
            value=5,
            help="Naikkan nilai ini kalau realtime terasa berat atau patah-patah.",
        )
        enable_realtime = st.checkbox(
            "Saya membuka halaman ini dari localhost atau HTTPS, tampilkan kamera realtime",
            value=False,
        )

        if enable_realtime:
            ctx = webrtc_streamer(
                key=f"emotion-realtime-{facing_mode}",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=RTC_CONFIGURATION,
                media_stream_constraints={
                    "video": {
                        "facingMode": facing_mode,
                        "width": {"ideal": 640},
                        "height": {"ideal": 480},
                    },
                    "audio": False,
                },
                video_html_attrs={
                    "autoPlay": True,
                    "controls": True,
                    "playsInline": True,
                    "muted": True,
                    "style": {"width": "100%"},
                },
                video_processor_factory=lambda: RealtimeEmotionProcessor(
                    model=model,
                    confidence=confidence,
                    yolo_mode=yolo_mode,
                    deepface_backend=deepface_backend,
                    frame_step=realtime_frame_step,
                ),
                async_processing=True,
            )

            if ctx.video_processor:
                ctx.video_processor.update_settings(
                    confidence=confidence,
                    yolo_mode=yolo_mode,
                    deepface_backend=deepface_backend,
                    frame_step=realtime_frame_step,
                )
                if st.button("Reset ringkasan realtime"):
                    ctx.video_processor.reset_counts()
                st.caption(
                    "Label emosi tampil langsung di video. Ringkasan di bawah diperbarui saat halaman rerun."
                )
                render_summary(ctx.video_processor.get_counts(), st.empty())
            else:
                st.info("Klik START pada komponen kamera untuk mulai realtime.")
        else:
            st.info(
                "Centang konfirmasi di atas hanya jika halaman sudah dibuka lewat localhost atau HTTPS."
            )


if __name__ == "__main__":
    main()
