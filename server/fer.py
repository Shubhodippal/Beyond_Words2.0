import streamlit as st
import cv2
import numpy as np 
from keras.models import model_from_json
from collections import Counter
import os
import json
import requests


HARDCODED_OPENAI_API_KEY = "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"
AI_REPORT_TONE = "professional"
AI_REPORT_DEPTH = "detailed"
AI_REPORT_MODEL = "gpt-5.4-2026-03-05"

json_file = open("emotiondetector.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("emotiondetector.h5")

haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}


def init_session_state():
    if 'processing_done' not in st.session_state:
        st.session_state.processing_done = False
    if 'uploaded_video_key' not in st.session_state:
        st.session_state.uploaded_video_key = None
    if 'emotion_counts' not in st.session_state:
        st.session_state.emotion_counts = {}
    if 'processed_frames' not in st.session_state:
        st.session_state.processed_frames = 0
    if 'emotion_sequence' not in st.session_state:
        st.session_state.emotion_sequence = []
    if 'show_analysis' not in st.session_state:
        st.session_state.show_analysis = False
    if 'ai_report' not in st.session_state:
        st.session_state.ai_report = ""


def get_openai_api_key():
    if HARDCODED_OPENAI_API_KEY:
        return HARDCODED_OPENAI_API_KEY

    env_key = os.getenv("OPENAI_API_KEY")
    if env_key:
        return env_key

    try:
        secret_key = st.secrets.get("OPENAI_API_KEY")
        if secret_key:
            return secret_key
    except Exception:
        return None

    return None


def build_timeline_summary(emotion_sequence, max_segments=10):
    if not emotion_sequence:
        return []

    segment_count = min(max_segments, len(emotion_sequence))
    indices = np.array_split(np.arange(len(emotion_sequence)), segment_count)
    timeline = []

    for idx, segment_indices in enumerate(indices, start=1):
        segment_emotions = [emotion_sequence[i] for i in segment_indices]
        segment_counter = Counter(segment_emotions)
        dominant_emotion, dominant_count = segment_counter.most_common(1)[0]
        timeline.append(
            {
                "segment": idx,
                "dominant_emotion": dominant_emotion,
                "dominant_share_percent": round((dominant_count / len(segment_emotions)) * 100, 2),
                "detections": len(segment_emotions),
            }
        )

    return timeline


def build_analysis_payload(emotion_counts, emotion_sequence, processed_frames):
    total_detections = sum(emotion_counts.values())
    percentages = {
        emotion: round((count / total_detections) * 100, 2)
        for emotion, count in emotion_counts.items()
    } if total_detections > 0 else {}

    transitions = Counter(
        f"{emotion_sequence[i]} -> {emotion_sequence[i + 1]}"
        for i in range(len(emotion_sequence) - 1)
    )
    top_transitions = [
        {"transition": transition, "count": count}
        for transition, count in transitions.most_common(8)
    ]

    timeline = build_timeline_summary(emotion_sequence)

    payload = {
        "processed_frames": processed_frames,
        "total_detections": total_detections,
        "detection_rate_percent": round((total_detections / processed_frames) * 100, 2) if processed_frames > 0 else 0,
        "dominant_emotion": max(emotion_counts, key=emotion_counts.get) if emotion_counts else None,
        "emotion_counts": emotion_counts,
        "emotion_percentages": percentages,
        "top_transitions": top_transitions,
        "timeline_summary": timeline,
    }
    return payload


def generate_openai_report(payload, api_key, report_tone, report_depth, model_name):
    prompt = (
        "You are an emotion analytics assistant. Create a detailed but practical report from the provided JSON. "
        "Use only this data, do not invent metrics. Avoid medical diagnosis language. "
        f"Tone: {report_tone}. Depth: {report_depth}. "
        "Return sections exactly in this order: Executive Summary, Key Findings, Temporal Pattern, "
        "Transition Insights, Caveats, Actionable Suggestions."
    )

    response = requests.post(
        "https://api.openai.com/v1/responses",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model_name,
            "input": [
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": prompt}],
                },
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": json.dumps(payload)}],
                },
            ],
            "temperature": 0.3,
        },
        timeout=90,
    )

    if response.status_code != 200:
        return None, f"OpenAI API error {response.status_code}: {response.text}"

    data = response.json()
    output_items = data.get("output", [])
    text_chunks = []
    for item in output_items:
        for content in item.get("content", []):
            if content.get("type") == "output_text" and content.get("text"):
                text_chunks.append(content["text"])

    if not text_chunks:
        return None, "OpenAI response did not include report text."

    return "\n\n".join(text_chunks).strip(), None


def show_analysis_result(emotion_counts, emotion_sequence, processed_frames):
    if not emotion_counts and not emotion_sequence:
        st.warning("No emotion data found to analyze.")
        return None

    total_detections = sum(emotion_counts.values())
    if total_detections == 0:
        st.warning("No faces were detected in the uploaded video.")
        return None

    analysis_payload = build_analysis_payload(emotion_counts, emotion_sequence, processed_frames)

    percentages = {
        emotion: (count / total_detections) * 100
        for emotion, count in emotion_counts.items()
    }

    st.subheader("Analysis Result")
    st.write(f"Total detections: {total_detections}")
    dominant_emotion = max(emotion_counts, key=emotion_counts.get)
    st.write(f"Dominant emotion: {dominant_emotion}")
    st.write(f"Detection rate: {analysis_payload['detection_rate_percent']}% of frames")

    result_rows = [
        {
            'emotion': emotion,
            'count': emotion_counts[emotion],
            'percentage': round(percentages[emotion], 2)
        }
        for emotion in emotion_counts
    ]
    st.dataframe(result_rows, use_container_width=True)

    chart_data = {
        'emotion': list(percentages.keys()),
        'percentage': list(percentages.values())
    }
    st.bar_chart(chart_data, x='emotion', y='percentage')

    st.subheader("Transition Insights")
    if analysis_payload['top_transitions']:
        st.dataframe(analysis_payload['top_transitions'], use_container_width=True)
    else:
        st.info("Not enough detections to calculate transitions.")

    st.subheader("Emotion Timeline (Segmented)")
    if analysis_payload['timeline_summary']:
        st.dataframe(analysis_payload['timeline_summary'], use_container_width=True)
    else:
        st.info("Not enough detections to build timeline summary.")

    return analysis_payload


def clear_analysis_data():
    st.session_state.processing_done = False
    st.session_state.uploaded_video_key = None
    st.session_state.emotion_counts = {}
    st.session_state.processed_frames = 0
    st.session_state.emotion_sequence = []
    st.session_state.show_analysis = False
    st.session_state.ai_report = ""

    if os.path.exists('output.txt'):
        os.remove('output.txt')
    if os.path.exists('temp_video.mp4'):
        os.remove('temp_video.mp4')

def detect_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        # Increase bounding box size by a factor of 1.5
        new_w = int(w * 1.5)
        new_h = int(h * 1.5)
        x_offset = int((new_w - w) / 2)  # Center the expanded box
        y_offset = int((new_h - h) / 2)
        #bounding box
        cv2.rectangle(image, (x - x_offset, y - y_offset), (x + new_w - x_offset, y + new_h - y_offset), (0, 255, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi_gray = roi_gray / 255.0
        roi_gray = np.reshape(roi_gray, (1, 48, 48, 1))
        return roi_gray

def main():
    st.title('Beyond Words')
    init_session_state()

    selected_video = st.file_uploader("Upload a video file", type=['mp4', 'avi'])

    if st.button('Delete Data'):
        clear_analysis_data()
        st.success("Temporary data and previous analysis have been cleared.")

    if selected_video is not None:
        current_video_key = f"{selected_video.name}-{selected_video.size}"
        if st.session_state.uploaded_video_key != current_video_key:
            st.session_state.uploaded_video_key = current_video_key
            st.session_state.processing_done = False
            st.session_state.emotion_counts = {}
            st.session_state.processed_frames = 0
            st.session_state.emotion_sequence = []
            st.session_state.show_analysis = False
            st.session_state.ai_report = ""

        with open('temp_video.mp4', 'wb') as f:
            f.write(selected_video.read())

        run_detection = st.button('Run')
        if run_detection:
            with open('output.txt', 'w') as file:
                file.write("")

            cap = cv2.VideoCapture('temp_video.mp4')
            progress_bar = st.progress(0)
            progress_status = st.empty()

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            processed_frames = 0
            detected_emotions = []

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                processed_frames += 1

                face = detect_faces(frame)
                if face is not None:
                    pred = model.predict(face)
                    emotion_label = labels[np.argmax(pred)]
                    detected_emotions.append(emotion_label)
                    with open('output.txt', 'a') as file:
                        file.write(f"{emotion_label}\n")

                if total_frames > 0:
                    progress_value = min(processed_frames / total_frames, 1.0)
                    progress_bar.progress(progress_value)
                    progress_status.text(f"Processing video: {processed_frames}/{total_frames} frames")
                else:
                    progress_status.text(f"Processing video: {processed_frames} frames")

            cap.release()

            st.session_state.processing_done = True
            st.session_state.processed_frames = processed_frames
            st.session_state.emotion_counts = dict(Counter(detected_emotions))
            st.session_state.emotion_sequence = detected_emotions
            st.session_state.show_analysis = False
            st.session_state.ai_report = ""

            progress_bar.progress(1.0)
            progress_status.text(f"Processing complete: {processed_frames} frames")
            st.success("Video processing completed. Click Analyze to view results.")

        analyze_result = st.button('Analyze', disabled=not st.session_state.processing_done)
        if analyze_result:
            st.session_state.show_analysis = True

        if st.session_state.show_analysis and st.session_state.processing_done:
            analysis_payload = show_analysis_result(
                st.session_state.emotion_counts,
                st.session_state.emotion_sequence,
                st.session_state.processed_frames,
            )

            if analysis_payload:
                st.subheader("AI Report (OpenAI)")
                key_from_env_or_secrets = get_openai_api_key()

                st.caption("AI report mode: professional • detailed • gpt-5.4-2026-03-05")

                generate_report = st.button("Generate AI Report")
                if generate_report:
                    api_key = key_from_env_or_secrets
                    if not api_key:
                        st.error("OpenAI API key is required. Set OPENAI_API_KEY or provide it above.")
                    else:
                        with st.spinner("Generating AI report..."):
                            report_text, report_error = generate_openai_report(
                                payload=analysis_payload,
                                api_key=api_key,
                                report_tone=AI_REPORT_TONE,
                                report_depth=AI_REPORT_DEPTH,
                                model_name=AI_REPORT_MODEL,
                            )
                        if report_error:
                            st.error(report_error)
                        else:
                            st.session_state.ai_report = report_text

                if st.session_state.ai_report:
                    st.markdown(st.session_state.ai_report)
                    st.download_button(
                        label="Download AI Report (.md)",
                        data=st.session_state.ai_report,
                        file_name="emotion_ai_report.md",
                        mime="text/markdown",
                    )

if __name__ == "__main__":
    main()