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

st.markdown(
    """
    <style>
    .main-title { font-size: 2.2rem; font-weight: 700; color: #1E293B; margin-bottom: 0.2rem; }
    .subtitle { font-size: 1.05rem; color: #64748B; margin-bottom: 1.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-title">Auto-Detecting 2D Character Animator</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload any drawing. Colors are automatically discovered, extracted, and rigged to black outlines without detaching.</div>', unsafe_allow_html=True)


def get_color_name(rgb):
    """Names common marker hues using classical HSV logic."""
    r, g, b = rgb
    hsv_pixel = cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2HSV)[0][0]
    hue, sat, val = hsv_pixel[0], hsv_pixel[1], hsv_pixel[2]

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


def extract_dominant_colors(image, num_clusters=6):
    """
    Uses OpenCV K-Means to find distinct accent colors,
    skipping black ink lines and white paper background.
    """
    # Downscale for fast clustering
    small = cv2.resize(image, (180, 180), interpolation=cv2.INTER_AREA)
    pixels = small.reshape((-1, 3)).astype(np.float32)

    # Filter out near-white/paper (>225 brightness) and dark ink lines (<50)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).flatten()
    valid_mask = (gray > 55) & (gray < 225)
    accent_pixels = pixels[valid_mask]

    if len(accent_pixels) < 100:
        return []

    # Apply K-Means Clustering (Classical math, no AI)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 15, 1.0)
    k = min(num_clusters, len(accent_pixels))
    _, labels, centers = cv2.kmeans(
        accent_pixels, k, None, criteria, 5, cv2.KMEANS_PP_CENTERS
    )

    # Compute coverage of each cluster
    counts = np.bincount(labels.flatten())
    total = np.sum(counts)

    detected = []
    for idx, center in enumerate(centers):
        b, g, r = int(center[0]), int(center[1]), int(center[2])
        coverage = (counts[idx] / total) * 100

        # Skip very tiny noisy artifacts (< 3% of colored pixels)
        if coverage < 3.0:
            continue

        hsv_c = cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2HSV)[0][0]
        # Skip unsaturated washed-out gray/beige paper tones
        if hsv_c[1] < 45:
            continue

        hex_code = f"#{r:02x}{g:02x}{b:02x}"
        name = get_color_name((r, g, b))

        detected.append({
            "name": f"{name} ({hex_code})",
            "bgr": (b, g, r),
            "hsv": hsv_c,
            "hex": hex_code,
            "coverage": coverage,
        })

    return detected


def segment_by_colors(image, chosen_colors):
    """Builds an elastic rig from selected color clusters."""
    h, w = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, black_borders = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY_INV)

    combined_mask = np.zeros((h, w), dtype=np.uint8)

    # Build mask for each selected color cluster with a small tolerance window
    for col in chosen_colors:
        h_val, s_val, v_val = col["hsv"]
        lower = np.array([max(0, int(h_val) - 15), max(30, int(s_val) - 60), max(30, int(v_val) - 60)])
        upper = np.array([min(180, int(h_val) + 15), 255, 255])

        mask = cv2.inRange(hsv, lower, upper)
        combined_mask = cv2.bitwise_or(combined_mask, mask)

    # Clean gaps
    kernel = np.ones((5, 5), np.uint8)
    clean_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

    # Connect to surrounding black pen outlines
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(clean_mask, connectivity=8)

    parts = []
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= 140:
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
    """Renders 3 motion styles with anchored joints (zero detaching)."""
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
            span = p["span"]
            base_y = p["base"][1]

            # Anchor formula: 0 at base, 1 at tip
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
                dx = step * (intensity * 1.3) * side * weight
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
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    original_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # 1. Automatic Palette Discovery (K-Means)
    with st.spinner("Analyzing image colors..."):
        detected_colors = extract_dominant_colors(original_img)

    st.sidebar.header("Auto-Discovered Colors")

    if not detected_colors:
        st.sidebar.warning("No clear accent marker colors found. The drawing might be pure black & white.")
        chosen_color_items = []
    else:
        # Display detected color options with preview badges
        color_options = {c["name"]: c for c in detected_colors}
        selected_names = st.sidebar.multiselect(
            "Select color(s) you want to animate:",
            options=list(color_options.keys()),
            default=[list(color_options.keys())[0]],  # default to first detected color
        )
        chosen_color_items = [color_options[name] for name in selected_names]

    st.sidebar.header("Motion Settings")
    motion_intensity = float(st.sidebar.slider("Motion Strength", 5.0, 30.0, 14.0))
    animation_speed = int(st.sidebar.slider("Frame Delay (ms)", 25, 90, 45))

    # 2. Segment & Rig Selected Parts
    detected_parts = segment_by_colors(original_img, chosen_color_items)

    # Visual Rig Preview
    debug_view = original_img.copy()
    for p in detected_parts:
        # Pinned Root (Red)
        cv2.circle(debug_view, (int(p["base"][0]), int(p["base"][1])), 7, (0, 0, 255), -1)
        # Elastic Tip (Green)
        cv2.circle(debug_view, (int(p["tip"][0]), int(p["tip"][1])), 5, (0, 255, 0), -1)
        # Skeleton Tension Line
        cv2.line(
            debug_view,
            (int(p["base"][0]), int(p["base"][1])),
            (int(p["tip"][0]), int(p["tip"][1])),
            (255, 0, 0),
            2,
        )

    # Preview Columns
    col_orig, col_pins = st.columns(2)
    with col_orig:
        st.subheader("Original Drawing")
        st.image(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB), use_container_width=True)

    with col_pins:
        st.subheader(f"Auto-Rigging ({len(detected_parts)} Parts Anchored)")
        st.image(cv2.cvtColor(debug_view, cv2.COLOR_BGR2RGB), use_container_width=True)
        st.caption("🔴 Red Dot = Fixed Joint (Pinned to Body) | 🟢 Green Dot = Elastic Tip")

    st.markdown("---")

    # 3. Animation Generation
    if len(detected_parts) == 0:
        st.info("Pick one or more detected colors from the left sidebar to activate limbs.")
    else:
        if st.button("✨ Generate 3 Animations", type="primary", use_container_width=True):
            with st.spinner("Calculating smooth elastic deformations..."):
                # 1. Natural Sway
                frames_sway = generate_animation_frames(
                    original_img, detected_parts, mode="sway", intensity=motion_intensity
                )
                gif_sway = build_gif_bytes(frames_sway, animation_speed)

                # 2. Playful Bounce
                frames_bounce = generate_animation_frames(
                    original_img, detected_parts, mode="bounce", intensity=motion_intensity
                )
                gif_bounce = build_gif_bytes(frames_bounce, animation_speed)

                # 3. Dynamic Wave
                frames_dyn = generate_animation_frames(
                    original_img, detected_parts, mode="dynamic", intensity=motion_intensity
                )
                gif_dyn = build_gif_bytes(frames_dyn, animation_speed)

            st.success("Animations rendered successfully!")

            # 3-Wide Display
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
