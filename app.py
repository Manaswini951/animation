import io
import math
import cv2
import numpy as np
import streamlit as st
from PIL import Image

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Hand-Drawn Character Animator",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🎨 Hand-Drawn Character Animator")
st.markdown(
    """
Upload a hand-drawn character and mark the parts you want to animate
using colored areas. The program anchors the joints to the body outlines
and generates attached, organic animations without AI.
"""
)


# ============================================================
# COLOR UTILITIES
# ============================================================

def get_color_name(rgb):
    """Convert RGB color into a human-readable name."""
    r, g, b = rgb
    pixel = np.uint8([[[b, g, r]]])
    hsv_pixel = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)[0][0]
    hue = int(hsv_pixel[0])
    sat = int(hsv_pixel[1])

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


def resize_for_processing(image, max_dimension=1200):
    """Resize only if image is unnecessarily large to preserve performance."""
    h, w = image.shape[:2]
    largest = max(h, w)
    if largest <= max_dimension:
        return image
    scale = max_dimension / largest
    return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


# ============================================================
# DOMINANT COLORS
# ============================================================

def extract_dominant_colors(image, num_clusters=7):
    try:
        small = cv2.resize(image, (180, 180), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        valid = (gray > 45) & (gray < 240)
        pixels = small[valid].astype(np.float32)

        if len(pixels) < 100:
            return []

        k = min(num_clusters, max(2, len(pixels) // 50))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 4, cv2.KMEANS_PP_CENTERS)

        counts = np.bincount(labels.flatten(), minlength=k)
        total = max(1, int(np.sum(counts)))
        detected = []

        for idx, center in enumerate(centers):
            b, g, r = [int(np.clip(x, 0, 255)) for x in center]
            coverage = (counts[idx] / total) * 100
            if coverage < 1.0:
                continue

            hsv = cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2HSV)[0][0]
            h_val, s_val, v_val = int(hsv[0]), int(hsv[1]), int(hsv[2])

            if s_val < 30:
                continue

            hex_code = f"#{r:02x}{g:02x}{b:02x}"
            detected.append({
                "label": f"{get_color_name((r, g, b))} ({hex_code}) {coverage:.1f}%",
                "hsv": (h_val, s_val, v_val),
                "hex": hex_code,
                "coverage": coverage,
            })

        detected.sort(key=lambda x: x["coverage"], reverse=True)
        return detected
    except Exception:
        return []


# ============================================================
# PART DETECTION & RIGGING
# ============================================================

def build_color_mask(image, chosen_colors):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    combined = np.zeros(hsv.shape[:2], dtype=np.uint8)

    for color in chosen_colors:
        h_val, s_val, v_val = color["hsv"]
        h_low = h_val - 15
        h_high = h_val + 15

        if h_low < 0:
            m1 = cv2.inRange(hsv, np.array([0, max(25, s_val - 80), max(25, v_val - 80)]), np.array([h_high, 255, 255]))
            m2 = cv2.inRange(hsv, np.array([180 + h_low, max(25, s_val - 80), max(25, v_val - 80)]), np.array([179, 255, 255]))
            mask = cv2.bitwise_or(m1, m2)
        elif h_high > 179:
            m1 = cv2.inRange(hsv, np.array([h_low, max(25, s_val - 80), max(25, v_val - 80)]), np.array([179, 255, 255]))
            m2 = cv2.inRange(hsv, np.array([0, max(25, s_val - 80), max(25, v_val - 80)]), np.array([h_high - 180, 255, 255]))
            mask = cv2.bitwise_or(m1, m2)
        else:
            mask = cv2.inRange(
                hsv,
                np.array([max(0, h_low), max(25, s_val - 80), max(25, v_val - 80)]),
                np.array([min(179, h_high), 255, 255]),
            )
        combined = cv2.bitwise_or(combined, mask)

    return combined


def detect_parts(image, chosen_colors):
    h, w = image.shape[:2]
    color_mask = build_color_mask(image, chosen_colors)

    mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    parts = []
    minimum_area = max(80, int(h * w * 0.00015))

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < minimum_area:
            continue

        component = (labels == i).astype(np.uint8) * 255
        ys, xs = np.where(component > 0)
        if len(xs) < 20:
            continue

        points = np.column_stack((xs, ys)).astype(np.float32)
        mean, eigenvectors = cv2.PCACompute(points, mean=None)
        center = mean[0]
        axis = eigenvectors[0]

        distances = np.dot(points - center, axis)
        min_point = points[np.argmin(distances)]
        max_point = points[np.argmax(distances)]

        # Base is closer to image center (torso/skull attachment point)
        image_center = np.array([w / 2.0, h / 2.0], dtype=np.float32)
        if np.linalg.norm(min_point - image_center) < np.linalg.norm(max_point - image_center):
            base, tip = min_point, max_point
        else:
            base, tip = max_point, min_point

        length = float(np.linalg.norm(tip - base))
        if length < 15:
            continue

        # Smooth expanded part mask
        expanded = cv2.dilate(component, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)

        parts.append({
            "full_mask": expanded > 0,
            "base": (float(base[0]), float(base[1])),
            "tip": (float(tip[0]), float(tip[1])),
            "center": (float(center[0]), float(center[1])),
            "length": length,
            "area": int(area),
        })

    parts.sort(key=lambda p: p["area"], reverse=True)
    return parts


# ============================================================
# SEAMLESS ELASTIC ANIMATION ENGINE (ZERO DETACHING)
# ============================================================

def animate_frame_elastic(original, parts, motion, frame_index, total_frames, intensity, smoothness=1.0):
    """
    Applies continuous mesh deformation: base joint has zero displacement
    (strictly attached to body), tip has maximum movement.
    """
    h, w = original.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = grid_x.astype(np.float32)
    map_y = grid_y.astype(np.float32)

    phase = 2.0 * math.pi * frame_index / total_frames
    sine = math.sin(phase)
    sine2 = math.sin(phase * 2.0 + math.pi / 4.0)

    for idx, part in enumerate(parts):
        mask = part["full_mask"]
        bx, by = part["base"]
        length = part["length"]

        # Calculate weight: 0 at base, 1 at tip
        dist = np.sqrt((grid_x - bx) ** 2 + (grid_y - by) ** 2)
        vert_weight = np.clip(dist / length, 0.0, 1.0)
        weight = np.power(vert_weight, 1.5) * smoothness

        direction = 1.0 if idx % 2 == 0 else -1.0

        if motion == "Sway":
            dx = sine * intensity * 0.8 * direction * weight
            dy = sine2 * intensity * 0.2 * weight
            map_x[mask] -= dx[mask]
            map_y[mask] -= dy[mask]

        elif motion == "Bounce":
            dy = sine * intensity * 0.6 * weight
            map_y[mask] -= dy[mask]

        elif motion == "Wave":
            dx = sine * intensity * 1.5 * direction * weight
            dy = (1.0 - abs(sine)) * intensity * 0.4 * weight
            map_x[mask] -= dx[mask]
            map_y[mask] -= dy[mask]

        elif motion == "Walk":
            dx = sine * intensity * 1.2 * direction * weight
            dy = sine2 * intensity * 0.4 * weight
            map_x[mask] -= dx[mask]
            map_y[mask] -= dy[mask]

        elif motion == "Wiggle":
            dx = sine * intensity * 0.7 * direction * weight
            map_x[mask] -= dx[mask]

    warped = cv2.remap(original, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return warped


def generate_animation_frames(original, parts, motion="Sway", intensity=12, frame_count=16, smoothness=1.0):
    frames = []
    for i in range(frame_count):
        frame = animate_frame_elastic(original, parts, motion, i, frame_count, intensity, smoothness)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(rgb))
    return frames


# ============================================================
# EXPORTERS
# ============================================================

def build_gif_bytes(frames, duration=60, loop=0):
    if not frames:
        return b""
    buffer = io.BytesIO()
    prepared = [f.convert("P", palette=Image.ADAPTIVE) for f in frames]
    prepared[0].save(
        buffer,
        format="GIF",
        save_all=True,
        append_images=prepared[1:],
        duration=duration,
        loop=loop,
        optimize=True,
    )
    return buffer.getvalue()


def create_rig_preview(image, parts):
    preview = image.copy()
    for idx, part in enumerate(parts):
        bx, by = int(part["base"][0]), int(part["base"][1])
        tx, ty = int(part["tip"][0]), int(part["tip"][1])
        cx, cy = int(part["center"][0]), int(part["center"][1])

        # Base (Anchor)
        cv2.circle(preview, (bx, by), 9, (0, 0, 255), -1)
        # Tip
        cv2.circle(preview, (tx, ty), 7, (0, 255, 0), -1)
        # Center
        cv2.circle(preview, (cx, cy), 5, (255, 0, 255), -1)
        # Link
        cv2.line(preview, (bx, by), (tx, ty), (255, 0, 0), 3)
        # Label
        cv2.putText(preview, f"PART {idx + 1}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(preview, f"PART {idx + 1}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return preview


# ============================================================
# APPLICATION MAIN
# ============================================================

if "animation_results" not in st.session_state:
    st.session_state.animation_results = {}

uploaded_file = st.file_uploader("📁 Upload your hand-drawn character", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    try:
        file_data = uploaded_file.getvalue()
        image_array = np.frombuffer(file_data, dtype=np.uint8)
        original_img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if original_img is None:
            st.error("Could not read this image.")
            st.stop()

        original_img = resize_for_processing(original_img, max_dimension=1400)

        # Sidebar setup
        st.sidebar.header("🎛️ Animation Controls")
        detected_colors = extract_dominant_colors(original_img)

        if detected_colors:
            st.sidebar.subheader("🎨 Detected Marking Colors")
            color_
