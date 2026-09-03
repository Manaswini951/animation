import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

st.set_page_config(page_title="Drawing Animator", layout="centered")
st.title("2D Drawing Part Animator (No AI)")

uploaded_file = st.file_uploader("Upload your character drawing", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 1. Read uploaded image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    h, w = img.shape[:2]

    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Original Drawing", use_container_width=True)

    # 2. Controls for the region and rotation
    st.sidebar.subheader("Area Selection (%)")
    ymin_pct = st.sidebar.slider("Top boundary (%)", 0, 100, 12)
    ymax_pct = st.sidebar.slider("Bottom boundary (%)", 0, 100, 32)
    xmin_pct = st.sidebar.slider("Left boundary (%)", 0, 100, 8)
    xmax_pct = st.sidebar.slider("Right boundary (%)", 0, 100, 22)

    st.sidebar.subheader("Motion Settings")
    swing_angle = st.sidebar.slider("Swing Angle (degrees)", 5, 45, 12)
    speed_duration = st.sidebar.slider("Frame Duration (ms)", 20, 100, 50)

    if st.button("Generate Animation"):
        with st.spinner("Processing limbs and generating frames..."):
            # Pixel conversions
            ymin, ymax = int(h * (ymin_pct / 100)), int(h * (ymax_pct / 100))
            xmin, xmax = int(w * (xmin_pct / 100)), int(w * (xmax_pct / 100))

            # Segment part inside ROI
            ear_mask = np.zeros((h, w), dtype=np.uint8)
            roi = img[ymin:ymax, xmin:xmax]
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray_roi, 240, 255, cv2.THRESH_BINARY_INV)
            ear_mask[ymin:ymax, xmin:xmax] = thresh

            # Clean mask
            kernel = np.ones((5, 5), np.uint8)
            ear_mask = cv2.morphologyEx(ear_mask, cv2.MORPH_CLOSE, kernel)

            # Joint/Pivot detection
            y_indices, x_indices = np.where(ear_mask > 0)
            if len(y_indices) == 0:
                st.error("No part detected inside the selected boundary. Adjust the sliders!")
            else:
                pivot_y = np.max(y_indices)
                pivot_x = int(np.mean(x_indices[y_indices == pivot_y]))

                # RGBA extraction & Inpainting
                b, g, r = cv2.split(img)
                ear_rgba = cv2.merge([b, g, r, ear_mask])
                clean_base = cv2.inpaint(img, ear_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)

                # Generate frames
                frames = []
                angles = np.sin(np.linspace(0, 2 * np.pi, 20)) * swing_angle

                for angle in angles:
                    rot_mat = cv2.getRotationMatrix2D((pivot_x, pivot_y), angle, 1.0)
                    rotated_part = cv2.warpAffine(ear_rgba, rot_mat, (w, h), flags=cv2.INTER_LINEAR)

                    frame = clean_base.copy()
                    alpha = rotated_part[:, :, 3] / 255.0

                    for c in range(3):
                        frame[:, :, c] = (1.0 - alpha) * frame[:, :, c] + alpha * rotated_part[:, :, c]

                    frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))

                # Save GIF to memory buffer
                gif_buffer = io.BytesIO()
                frames[0].save(
                    gif_buffer,
                    format="GIF",
                    save_all=True,
                    append_images=frames[1:],
                    duration=speed_duration,
                    loop=0
                )
                gif_bytes = gif_buffer.getvalue()

                # Display in Streamlit
                st.subheader("Animated Result")
                st.image(gif_bytes)
                st.download_button("Download GIF", data=gif_bytes, file_name="animated_drawing.gif", mime="image/gif")
