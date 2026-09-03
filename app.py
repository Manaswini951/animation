import io
import cv2
import numpy as np
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Auto Pink-Part Animator", layout="wide")
st.title("Auto Color-Targeted 2D Animator (No AI)")

uploaded_file = st.file_uploader(
    "Upload your hand-drawn character", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # 1. Load image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    h, w = img.shape[:2]

    # Sidebar: Fine-tune the pink range if needed
    st.sidebar.header("Color Sensitivity (Dark Pink/Magenta)")
    h_min = st.sidebar.slider("Hue Min", 0, 180, 135)
    h_max = st.sidebar.slider("Hue Max", 0, 180, 175)
    s_min = st.sidebar.slider("Saturation Min", 0, 255, 60)
    v_min = st.sidebar.slider("Value/Brightness Min", 0, 255, 60)

    st.sidebar.header("Animation Settings")
    swing_angle = float(st.sidebar.slider("Swing Angle (degrees)", 2, 35, 12))
    speed_duration = int(st.sidebar.slider("Frame Duration (ms)", 20, 120, 50))

    # 2. Convert to HSV & Create Mask for Dark Pink/Magenta
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_pink = np.array([h_min, s_min, v_min])
    upper_pink = np.array([h_max, 255, 255])
    raw_mask = cv2.inRange(hsv, lower_pink, upper_pink)

    # Clean noise and bridge marker stroke gaps
    kernel = np.ones((5, 5), np.uint8)
    clean_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, kernel)

    # 3. Find distinct parts (Left ear, Right ear, etc.)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        clean_mask, connectivity=8
    )

    # Filter out tiny noise specks (require at least 150 pixels)
    valid_parts = []
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= 150:
            part_mask = (labels == i).astype(np.uint8) * 255
            # Joint Pivot = lowest point (base of the ear)
            y_pts, x_pts = np.where(part_mask > 0)
            pivot_y = float(np.max(y_pts))
            pivot_x = float(np.mean(x_pts[y_pts == int(pivot_y)]))
            valid_parts.append({
                "mask": part_mask,
                "pivot": (pivot_x, pivot_y)
            })

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Auto-Detected Pink Zones")
        preview_mask = cv2.cvtColor(clean_mask, cv2.COLOR_GRAY2RGB)
        # Draw green circles on detected pivot joints
        for part in valid_parts:
            cv2.circle(
                preview_mask,
                (int(part["pivot"][0]), int(part["pivot"][1])),
                7,
                (0, 255, 0),
                -1,
            )
        st.image(
            preview_mask,
            caption=f"Detected {len(valid_parts)} pink limb(s). Green dots = pivot joints.",
            use_container_width=True,
        )

    with col2:
        st.subheader("Animated Output")
        if len(valid_parts) == 0:
            st.warning("No dark pink regions detected. Try adjusting the Hue/Saturation sliders.")
        else:
            if st.button("Generate Animation", type="primary"):
                with st.spinner("Isolating parts and rendering frames..."):
                    # Extract each part as an independent RGBA layer
                    b, g, r = cv2.split(img)
                    part_layers = []
                    total_mask = np.zeros((h, w), dtype=np.uint8)

                    for part in valid_parts:
                        p_mask = part["mask"]
                        total_mask = cv2.bitwise_or(total_mask, p_mask)
                        p_rgba = cv2.merge([b, g, r, p_mask])
                        part_layers.append({
                            "rgba": p_rgba,
                            "pivot": part["pivot"]
                        })

                    # Inpaint the base image so background stays clean behind moving parts
                    clean_base = cv2.inpaint(
                        img, total_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA
                    )

                    # Build animation sequence
                    frames = []
                    angle_steps = np.sin(np.linspace(0, 2 * np.pi, 20))

                    for step in angle_steps:
                        frame = clean_base.copy()

                        # Animate all parts
                        for idx, part in enumerate(part_layers):
                            # Alternate swing direction for symmetric ear twitching
                            direction = 1.0 if idx % 2 == 0 else -1.0
                            current_angle = float(step * swing_angle * direction)

                            rot_mat = cv2.getRotationMatrix2D(
                                part["pivot"], current_angle, 1.0
                            )
                            rotated = cv2.warpAffine(
                                part["rgba"],
                                rot_mat,
                                (w, h),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT,
                            )

                            # Blend onto frame
                            alpha = rotated[:, :, 3] / 255.0
                            for c in range(3):
                                frame[:, :, c] = (
                                    (1.0 - alpha) * frame[:, :, c]
                                    + alpha * rotated[:, :, c]
                                )

                        frames.append(
                            Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        )

                    # Save as GIF
                    gif_buffer = io.BytesIO()
                    frames[0].save(
                        gif_buffer,
                        format="GIF",
                        save_all=True,
                        append_images=frames[1:],
                        duration=speed_duration,
                        loop=0,
                    )
                    gif_bytes = gif_buffer.getvalue()

                    st.image(gif_bytes, caption="Wiggling Pink Parts")
                    st.download_button(
                        label="Download GIF",
                        data=gif_bytes,
                        file_name="animated_ears.gif",
                        mime="image/gif",
                    )
