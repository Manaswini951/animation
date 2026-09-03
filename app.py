import io
import cv2
import numpy as np
import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="2D Hand-Drawn Character Animator",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished interface
st.markdown(
    """
    <style>
    .main-title { font-size: 2.2rem; font-weight: 700; color: #1E293B; margin-bottom: 0.2rem; }
    .subtitle { font-size: 1.05rem; color: #64748B; margin-bottom: 1.5rem; }
    .card { background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 10px; padding: 16px; margin-bottom: 16px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-title">2D Character Animator (Pure Computer Vision)</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Select your hand-drawn color targets. The script anchors joints to the black outlines to generate 3 unique animations without tearing.</div>', unsafe_allow_html=True)

# Preset HSV definitions for common colored markers
COLOR_PROFILES = {
    "Pink / Magenta (e.g. Ears)": {
        "lower": np.array([135, 60, 60]),
        "upper": np.array([175, 255, 255]),
    },
    "Blue / Violet (e.g. Hooves / Tail / Eyes)": {
        "lower": np.array([100, 70, 50]),
        "upper": np.array([135, 255, 255]),
    },
    "Red / Orange (e.g. Muzzle / Spots)": {
        "lower": np.array([0, 90, 80]),
        "upper": np.array([15, 255, 255]),
    },
    "Green (Accent markings)": {
        "lower": np.array([35, 60, 50]),
        "upper": np.array([85, 255, 255]),
    },
}

uploaded_file = st.file_uploader(
    "Upload hand-drawn character image", type=["jpg", "jpeg", "png"]
)

# Sidebar settings
st.sidebar.header("1. Target Limbs & Parts")
selected_targets = st.sidebar.multiselect(
    "Choose parts to animate by color:",
    options=list(COLOR_PROFILES.keys()),
    default=["Pink / Magenta (e.g. Ears)"],
)

st.sidebar.header("2. Motion Dynamics")
motion_intensity = float(st.sidebar.slider("Movement Range", 5.0, 30.0, 14.0))
animation_speed = int(st.sidebar.slider("Frame Interval (ms)", 25, 90, 45))


def extract_bordered_parts(image, target_names):
    """
    Extracts colored sections and expands them smoothly to the enclosing black outline.
    """
    h, w = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect black pen/marker outlines
    _, black_borders = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY_INV)

    combined_color_mask = np.zeros((h, w), dtype=np.uint8)
    for name in target_names:
        bounds = COLOR_PROFILES[name]
        mask = cv2.inRange(hsv, bounds["lower"], bounds["upper"])
        combined_color_mask = cv2.bitwise_or(combined_color_mask, mask)

    # Clean small color gaps
    kernel = np.ones((5, 5), np.uint8)
    clean_color_mask = cv2.morphologyEx(combined_color_mask, cv2.MORPH_CLOSE, kernel)

    # Connected components for each distinct limb/part
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(clean_color_mask, connectivity=8)

    parts = []
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= 120:  # Ignore tiny noise flecks
            single_part_seed = (labels == i).astype(np.uint8) * 255

            # Dilate color zone to connect tightly with outer black boundary
            dilated = cv2.dilate(single_part_seed, np.ones((7, 7), np.uint8), iterations=2)
            part_mask = cv2.bitwise_and(dilated, cv2.bitwise_not(black_borders))
            part_mask = cv2.bitwise_or(part_mask, single_part_seed)

            y_pts, x_pts = np.where(part_mask > 0)
            if len(y_pts) == 0:
                continue

            # Anchor = base joint (lowest Y value where part meets body)
            base_y = float(np.max(y_pts))
            base_x = float(np.mean(x_pts[y_pts == int(base_y)]))

            # Tip = point that moves the most (highest point)
            tip_y = float(np.min(y_pts))
            tip_x = float(np.mean(x_pts[y_pts == int(tip_y)]))

            span = max(1.0, base_y - tip_y)

            parts.append({
                "mask": (part_mask > 0),
                "base": (base_x, base_y),
                "tip": (tip_x, tip_y),
                "span": span,
            })

    return parts, clean_color_mask


def generate_animation_frames(img, parts, mode="sway", intensity=12.0):
    """
    Renders elastic mesh deformation anchored at base joints to avoid limb detachment.
    """
    h, w = img.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    grid_x = grid_x.astype(np.float32)
    grid_y = grid_y.astype(np.float32)

    num_frames = 22
    t_steps = np.sin(np.linspace(0, 2 * np.pi, num_frames))
    frames = []

    for step in t_steps:
        map_x = grid_x.copy()
        map_y = grid_y.copy()

        for idx, p in enumerate(parts):
            mask = p["mask"]
            base_x, base_y = p["base"]
            span = p["span"]

            # Anchor equation: weight = 0 at base (pinned), weight = 1 at tip
            vert_dist = np.clip((base_y - grid_y) / span, 0.0, 1.0)
            weight = np.power(vert_dist, 1.5)

            # Alternate swing directions for symmetry
            side = 1.0 if idx % 2 == 0 else -1.0

            if mode == "sway":
                # Natural horizontal swaying
                dx = step * intensity * side * weight
                map_x[mask] -= dx[mask]

            elif mode == "bounce":
                # Vertical squash & stretch
                dy = step * (intensity * 0.7) * weight
                map_y[mask] -= dy[mask]

            elif mode == "dynamic":
                # Coupled horizontal & vertical arc movement
                dx = step * (intensity * 1.3) * side * weight
                dy = (1.0 - abs(step)) * (intensity * 0.4) * weight
                map_x[mask] -= dx[mask]
                map_y[mask] -= dy[mask]

        # Interpolate frame without leaving any empty seams
        warped = cv2.remap(
            img,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )
        frames.append(Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)))

    return frames


def build_gif_bytes(frames, duration):
    buf = io.BytesIO()
    frames[0].save(
        buf,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
    )
    return buf.getvalue()


if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    original_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Process detections
    detected_parts, mask_preview = extract_bordered_parts(original_img, selected_targets)

    # Diagnostic preview with anchored pins
    debug_view = original_img.copy()
    for p in detected_parts:
        # Red Pin: Root Joint (Never moves)
        cv2.circle(debug_view, (int(p["base"][0]), int(p["base"][1])), 7, (0, 0, 255), -1)
        # Green Pin: Moving Tip
        cv2.circle(debug_view, (int(p["tip"][0]), int(p["tip"][1])), 5, (0, 255, 0), -1)
        # Tension Link
        cv2.line(
            debug_view,
            (int(p["base"][0]), int(p["base"][1])),
            (int(p["tip"][0]), int(p["tip"][1])),
            (255, 0, 0),
            2,
        )

    # Top Section: Drawing Inspection
    col_orig, col_pins = st.columns(2)
    with col_orig:
        st.subheader("Original Drawing")
        st.image(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB), use_container_width=True)

    with col_pins:
        st.subheader(f"Joint Detection ({len(detected_parts)} Limb/Part Regions Found)")
        st.image(cv2.cvtColor(debug_view, cv2.COLOR_BGR2RGB), use_container_width=True)
        st.caption("🔴 Red Dot = Fixed Joint (Pinned to Body) | 🟢 Green Dot = Elastic Tip")

    st.markdown("---")

    if len(detected_parts) == 0:
        st.warning("No matching color regions found. Select additional colors from the left sidebar.")
    else:
        if st.button("✨ Generate 3 Distinct Animations", type="primary", use_container_width=True):
            with st.spinner("Calculating non-linear mesh deformation and rendering 3 GIF sequences..."):
                # 1. Sway / Twitch
                frames_sway = generate_animation_frames(
                    original_img, detected_parts, mode="sway", intensity=motion_intensity
                )
                gif_sway = build_gif_bytes(frames_sway, animation_speed)

                # 2. Bounce / Stretch
                frames_bounce = generate_animation_frames(
                    original_img, detected_parts, mode="bounce", intensity=motion_intensity
                )
                gif_bounce = build_gif_bytes(frames_bounce, animation_speed)

                # 3. Dynamic Arc / Waggle
                frames_dyn = generate_animation_frames(
                    original_img, detected_parts, mode="dynamic", intensity=motion_intensity
                )
                gif_dyn = build_gif_bytes(frames_dyn, animation_speed)

            st.success("All 3 animations rendered successfully!")

            # Display 3-wide animation showcase
            anim1, anim2, anim3 = st.columns(3)

            with anim1:
                st.markdown("### 1. Natural Sway")
                st.image(gif_sway, use_container_width=True)
                st.download_button(
                    "Download Sway GIF",
                    gif_sway,
                    "animation_sway.gif",
                    "image/gif",
                    use_container_width=True,
                )

            with anim2:
                st.markdown("### 2. Playful Bounce")
                st.image(gif_bounce, use_container_width=True)
                st.download_button(
                    "Download Bounce GIF",
                    gif_bounce,
                    "animation_bounce.gif",
                    "image/gif",
                    use_container_width=True,
                )

            with anim3:
                st.markdown("### 3. Dynamic Wave")
                st.image(gif_dyn, use_container_width=True)
                st.download_button(
                    "Download Wave GIF",
                    gif_dyn,
                    "animation_dynamic.gif",
                    "image/gif",
                    use_container_width=True,
                )
