import io
import math
import cv2
import numpy as np
import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Hand-Drawn Character Animator",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🎨 Hand-Drawn Character Animator")
st.markdown("Upload your drawing. Marked colored areas are anchored to the body without tearing or detaching.")


def get_color_name(rgb):
    r, g, b = rgb
    pixel = np.uint8([[[b, g, r]]])
    hsv = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)[0][0]
    hue, sat = int(hsv[0]), int(hsv[1])

    if sat < 35:
        return "Neutral"
    if hue < 8 or hue >= 172:
        return "Red"
    elif 8 <= hue < 22:
        return "Orange"
    elif 22 <= hue < 35:
        return "Yellow"
    elif 35 <= hue < 85:
        return "Green"
    elif 85 <= hue < 130:
        return "Blue"
    elif 130 <= hue < 155:
        return "Purple"
    elif 155 <= hue < 172:
        return "Pink"
    return "Accent"


def extract_dominant_colors(image, num_clusters=5):
    try:
        small = cv2.resize(image, (120, 120), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        valid = (gray > 45) & (gray < 235)
        pixels = small[valid].reshape(-1, 3).astype(np.float32)

        if len(pixels) < 50:
            return []

        k = min(num_clusters, max(1, len(pixels) // 20))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 15, 1.0)
        _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)

        counts = np.bincount(labels.flatten(), minlength=k)
        total = max(1, int(np.sum(counts)))
        detected = []

        for idx, center in enumerate(centers):
            b, g, r = [int(np.clip(x, 0, 255)) for x in center]
            coverage = (counts[idx] / total) * 100
            if coverage < 1.5:
                continue

            hsv = cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2HSV)[0][0]
            if int(hsv[1]) < 30:
                continue

            hex_code = f"#{r:02x}{g:02x}{b:02x}"
            detected.append({
                "label": f"{get_color_name((r, g, b))} ({hex_code}) {coverage:.1f}%",
                "hsv": (int(hsv[0]), int(hsv[1]), int(hsv[2])),
                "hex": hex_code,
            })

        return detected
    except Exception:
        return []


def detect_parts_safe(image, chosen_colors):
    h, w = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    combined = np.zeros((h, w), dtype=np.uint8)

    for color in chosen_colors:
        h_val, s_val, v_val = color["hsv"]
        lower = np.array([max(0, h_val - 16), max(25, s_val - 70), max(25, v_val - 70)])
        upper = np.array([min(179, h_val + 16), 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
        combined = cv2.bitwise_or(combined, mask)

    clean_mask = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(clean_mask, connectivity=8)

    parts = []
    min_area = max(60, int(h * w * 0.0001))

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue

        comp = (labels == i).astype(np.uint8) * 255
        ys, xs = np.where(comp > 0)
        if len(xs) < 15:
            continue

        # Safe Joint Detection using Min/Max Bounds
        base_y = float(np.max(ys))
        base_x = float(np.mean(xs[ys == int(base_y)]))

        tip_y = float(np.min(ys))
        tip_x = float(np.mean(xs[ys == int(tip_y)]))

        length = max(10.0, float(math.hypot(tip_x - base_x, tip_y - base_y)))
        expanded = cv2.dilate(comp, np.ones((5, 5), np.uint8), iterations=1)

        parts.append({
            "mask": expanded > 0,
            "base": (base_x, base_y),
            "tip": (tip_x, tip_y),
            "center": (float(np.mean(xs)), float(np.mean(ys))),
            "length": length,
            "area": int(area),
        })

    parts.sort(key=lambda p: p["area"], reverse=True)
    return parts


def animate_frame_elastic(original, parts, motion, frame_index, total_frames, intensity):
    h, w = original.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = grid_x.astype(np.float32)
    map_y = grid_y.astype(np.float32)

    phase = 2.0 * math.pi * frame_index / total_frames
    sine = math.sin(phase)

    for idx, part in enumerate(parts):
        mask = part["mask"]
        bx, by = part["base"]
        length = part["length"]

        dist = np.sqrt((grid_x - bx) ** 2 + (grid_y - by) ** 2)
        weight = np.power(np.clip(dist / length, 0.0, 1.0), 1.5)
        dir_mult = 1.0 if idx % 2 == 0 else -1.0

        if motion == "Sway":
            dx = sine * intensity * dir_mult * weight
            map_x[mask] -= dx[mask]
        elif motion == "Bounce":
            dy = sine * (intensity * 0.7) * weight
            map_y[mask] -= dy[mask]
        elif motion == "Wave":
            dx = sine * (intensity * 1.4) * dir_mult * weight
            dy = (1.0 - abs(sine)) * (intensity * 0.4) * weight
            map_x[mask] -= dx[mask]
            map_y[mask] -= dy[mask]

    return cv2.remap(original, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def build_gif(frames, duration=60):
    buf = io.BytesIO()
    prepared = [f.convert("P", palette=Image.ADAPTIVE) for f in frames]
    prepared[0].save(buf, format="GIF", save_all=True, append_images=prepared[1:], duration=duration, loop=0)
    return buf.getvalue()


# --- Main UI ---
uploaded_file = st.file_uploader("Upload hand-drawn character", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    try:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img is None:
            st.error("Invalid image file.")
            st.stop()

        # Resize if massive
        h, w = img.shape[:2]
        if max(h, w) > 1200:
            scale = 1200.0 / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        detected_colors = extract_dominant_colors(img)

        st.sidebar.header("Controls")
        chosen_colors = []
        if detected_colors:
            color_dict = {c["label"]: c for c in detected_colors}
            selected_labels = st.sidebar.multiselect("Colors to Animate", list(color_dict.keys()), default=[list(color_dict.keys())[0]])
            chosen_colors = [color_dict[lbl] for lbl in selected_labels]
        else:
            st.sidebar.info("No marker colors detected.")

        motion = st.sidebar.selectbox("Motion Style", ["Sway", "Bounce", "Wave"])
        intensity = float(st.sidebar.slider("Motion Strength", 4, 30, 12))

        parts = detect_parts_safe(img, chosen_colors) if chosen_colors else []

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original")
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)

        with col2:
            st.subheader(f"Rigged Parts ({len(parts)} found)")
            preview = img.copy()
            for p in parts:
                cv2.circle(preview, (int(p["base"][0]), int(p["base"][1])), 7, (0, 0, 255), -1)
                cv2.circle(preview, (int(p["tip"][0]), int(p["tip"][1])), 5, (0, 255, 0), -1)
                cv2.line(preview, (int(p["base"][0]), int(p["base"][1])), (int(p["tip"][0]), int(p["tip"][1])), (255, 0, 0), 2)
            st.image(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB), use_container_width=True)

        st.markdown("---")
        if st.button("✨ Generate Animation", type="primary", use_container_width=True, disabled=len(parts) == 0):
            with st.spinner("Rendering seamless animation..."):
                frames = []
                for i in range(16):
                    warped = animate_frame_elastic(img, parts, motion, i, 16, intensity)
                    frames.append(Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)))

                gif_data = build_gif(frames, duration=50)
                st.subheader(f"Result: {motion}")
                st.image(gif_data, use_container_width=True)
                st.download_button("Download GIF", gif_data, f"{motion.lower()}_animation.gif", "image/gif")

    except Exception as e:
        st.error(f"Error: {e}")
