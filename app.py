import io
import cv2
import numpy as np
import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Auto-Color 2D Animator",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("2D Hand-Drawn Character Animator")
st.markdown("Upload any drawing. Limbs and colored areas are automatically identified and anchored to avoid detaching.")


def get_color_name(rgb):
    """Categorizes color hue using standard HSV boundaries."""
    r, g, b = rgb
    pixel = np.uint8([[[b, g, r]]])
    hsv_pixel = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)[0][0]
    hue, sat = int(hsv_pixel[0]), int(hsv_pixel[1])

    if sat < 40:
        return "Neutral / Gray"
    if hue < 10 or hue >= 170:
        return "Red"
    elif 10 <= hue < 25:
        return "Orange / Tan"
    elif 25 <= hue < 35:
        return "Yellow"
    elif 35 <= hue < 85:
        return "Green"
    elif 85 <= hue < 130:
        return "Blue"
    elif 130 <= hue < 155:
        return "Purple / Violet"
    elif 155 <= hue < 170:
        return "Pink / Magenta"
    return "Accent Color"


def extract_dominant_colors(image, num_clusters=5):
    """Safely extracts dominant marker colors using KMeans."""
    try:
        small = cv2.resize(image, (150, 150), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        
        # Exclude black ink lines and white/cream paper background
        valid_pixels = (gray > 50) & (gray < 225)
        accent_pixels = small[valid_pixels].reshape(-1, 3).astype(np.float32)

        if len(accent_pixels) < 50:
            return []

        k = min(num_clusters, max(1, len(accent_pixels) // 10))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 15, 1.0)
        _, labels, centers = cv2.kmeans(
            accent_pixels, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS
        )

        counts = np.bincount(labels.flatten())
        total = max(1, np.sum(counts))

        detected = []
        for idx, center in enumerate(centers):
            b, g, r = int(center[0]), int(center[1]), int(center[2])
            coverage = float((counts[idx] / total) * 100)

            if coverage < 2.0:
                continue

            hsv_c = cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2HSV)[0][0]
            if int(hsv_c[1]) < 35:  # Skip faded grays
                continue

            hex_code = f"#{r:02x}{g:02x}{b:02x}"
            name = get_color_name((r, g, b))

            detected.append({
                "label": f"{name} ({hex_code})",
                "hsv": (int(hsv_c[0]), int(hsv_c[1]), int(hsv_c[2])),
                "hex": hex_code,
            })

        return detected
    except Exception:
        return []


def segment_by_colors(image, chosen_colors):
    """Extracts limb regions anchored against surrounding ink borders."""
    h, w = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, black_borders = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY_INV)
    combined_mask = np.zeros((h, w), dtype=np.uint8)

    for col in chosen_colors:
        h_val, s_val, v_val = col["hsv"]
        lower = np.array([max(0, h_val - 18), max(30, s_val - 70), max(30, v_val - 70)])
        upper = np.array([min(180, h_val + 18), 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
        combined_mask = cv2.bitwise_or(combined_mask, mask)

    kernel = np.ones((5, 5), np.uint8)
    clean_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(clean_mask, connectivity=8)

    parts = []
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= 100:
            seed = (labels == i).astype(np.uint8) * 255
            dilated = cv2.dilate(seed, np.ones((7, 7), np.uint8), iterations=2)
            part_mask = cv2.bitwise_and(dilated, cv2.bitwise_not(black_borders))
            part_mask = cv2.bitwise_or(part_mask, seed)

            y_pts, x_pts = np.where(part_mask > 0)
            if len(y_pts) == 0:
                continue

            base_y = float(np.max(y_pts))
            base_x = float(np.mean(x_pts[y_pts == int(base_y)]))
            tip_y = float(np.min(y_pts))
            tip_x = float(np.mean(x_pts[y_pts == int(tip_y)]))

            parts.append({
                "mask": (part_mask > 0),
                "base": (base_x, base_y),
                "tip": (tip_x, tip_y),
                "span": max(1.0, base_y - tip_y),
            })

    return parts


def generate_animation_frames(img, parts, mode="sway", intensity=14.0):
    """Generates frames using elastic warp pinned at the root."""
    h, w = img.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    grid_x = grid_x.astype(np.float32)
    grid_y = grid_y.astype(np.float32)

    num_frames = 20
    t_steps = np.sin(np.linspace(0, 2 * np.pi, num_frames))
    frames = []

    for step in t_steps:
        map_x = grid_x.copy()
        map_y = grid_y.copy()

        for idx, p in enumerate(parts):
            mask = p["mask"]
            span = p["span"]
            base_y = p["base"][1]

            vert_dist = np.clip((base_y - grid_y) / span, 0.0, 1.0)
            weight = np.power(vert_dist, 1.5)
            side = 1.0 if idx % 2 == 0 else -1.0

            if mode == "sway":
                dx = step * intensity * side * weight
                map_x[mask] -= dx[mask]
            elif mode == "bounce":
                dy = step * (intensity * 0.7) * weight
                map_y[mask] -= dy[mask]
            elif mode == "dynamic":
                dx = step * (intensity * 1.2) * side * weight
                dy = (1.0 - abs(step)) * (intensity * 0.4) * weight
                map_x[mask] -= dx[mask]
                map_y[mask] -= dy[mask]

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


uploaded_file = st.file_uploader(
    "Upload hand-drawn character image", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    try:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        original_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if original_img is None:
            st.error("Could not read uploaded image. Please try another file.")
            st.stop()

        # Extract colors
        detected_colors = extract_dominant_colors(original_img)

        st.sidebar.header("Detected Colors")
        chosen_color_items = []

        if detected_colors:
            color_options = {c["label"]: c for c in detected_colors}
            selected_labels = st.sidebar.multiselect(
                "Parts to animate:",
                options=list(color_options.keys()),
                default=[list(color_options.keys())[0]],
            )
            chosen_color_items = [color_options[lbl] for lbl in selected_labels]
        else:
            st.sidebar.info("No accent colors detected. Using default pink marker detection.")
            chosen_color_items = [{
                "label": "Default Pink",
                "hsv": (155, 120, 120),
                "hex": "#FF1493",
            }]

        motion_intensity = float(st.sidebar.slider("Motion Strength", 5.0, 30.0, 14.0))
        animation_speed = int(st.sidebar.slider("Frame Interval (ms)", 25, 90, 45))

        detected_parts = segment_by_colors(original_img, chosen_color_items)

        # Visual preview with anchor joints
        debug_view = original_img.copy()
        for p in detected_parts:
            cv2.circle(debug_view, (int(p["base"][0]), int(p["base"][1])), 7, (0, 0, 255), -1)
            cv2.circle(debug_view, (int(p["tip"][0]), int(p["tip"][1])), 5, (0, 255, 0), -1)
            cv2.line(
                debug_view,
                (int(p["base"][0]), int(p["base"][1])),
                (int(p["tip"][0]), int(p["tip"][1])),
                (255, 0, 0),
                2,
            )

        col_orig, col_pins = st.columns(2)
        with col_orig:
            st.subheader("Original Drawing")
            st.image(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB), use_container_width=True)

        with col_pins:
            st.subheader(f"Rig Preview ({len(detected_parts)} Parts Anchored)")
            st.image(cv2.cvtColor(debug_view, cv2.COLOR_BGR2RGB), use_container_width=True)
            st.caption("🔴 Red Dot = Joint Anchor (Stationary) | 🟢 Green Dot = Elastic Tip")

        st.markdown("---")

        if len(detected_parts) == 0:
            st.warning("No marked color zones found inside boundaries. Select a different color in the sidebar.")
        else:
            if st.button("✨ Generate 3 Animations", type="primary", use_container_width=True):
                with st.spinner("Rendering 3 animations..."):
                    frames_sway = generate_animation_frames(
                        original_img, detected_parts, mode="sway", intensity=motion_intensity
                    )
                    gif_sway = build_gif_bytes(frames_sway, animation_speed)

                    frames_bounce = generate_animation_frames(
                        original_img, detected_parts, mode="bounce", intensity=motion_intensity
                    )
                    gif_bounce = build_gif_bytes(frames_bounce, animation_speed)

                    frames_dyn = generate_animation_frames(
                        original_img, detected_parts, mode="dynamic", intensity=motion_intensity
                    )
                    gif_dyn = build_gif_bytes(frames_dyn, animation_speed)

                st.success("Render complete!")

                anim1, anim2, anim3 = st.columns(3)
                with anim1:
                    st.markdown("### 1. Natural Sway")
                    st.image(gif_sway, use_container_width=True)
                    st.download_button("Download Sway GIF", gif_sway, "sway.gif", "image/gif", use_container_width=True)

                with anim2:
                    st.markdown("### 2. Playful Bounce")
                    st.image(gif_bounce, use_container_width=True)
                    st.download_button("Download Bounce GIF", gif_bounce, "bounce.gif", "image/gif", use_container_width=True)

                with anim3:
                    st.markdown("### 3. Dynamic Wave")
                    st.image(gif_dyn, use_container_width=True)
                    st.download_button("Download Wave GIF", gif_dyn, "wave.gif", "image/gif", use_container_width=True)

    except Exception as e:
        st.error(f"Execution Error: {e}")
