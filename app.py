import io
import cv2
import numpy as np
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Drawing Animator", layout="wide")
st.title("2D Drawing Part Animator (No AI)")

uploaded_file = st.file_uploader(
    "Upload your character drawing", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # 1. Read uploaded image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    h, w = img.shape[:2]

    # Sidebar Controls
    st.sidebar.header("1. Selection Area (%)")
    ymin_pct = st.sidebar.slider("Top boundary (%)", 0, 100, 12)
    ymax_pct = st.sidebar.slider("Bottom boundary (%)", 0, 100, 32)
    xmin_pct = st.sidebar.slider("Left boundary (%)", 0, 100, 8)
    xmax_pct = st.sidebar.slider("Right boundary (%)", 0, 100, 22)

    # Convert percentages to actual pixel values
    ymin = int(h * (min(ymin_pct, ymax_pct) / 100))
    ymax = int(h * (max(ymin_pct, ymax_pct) / 100))
    xmin = int(w * (min(xmin_pct, xmax_pct) / 100))
    xmax = int(w * (max(xmin_pct, xmax_pct) / 100))

    st.sidebar.header("2. Motion Controls")
    swing_angle = float(st.sidebar.slider("Swing Angle (degrees)", 2, 45, 12))
    speed_duration = int(st.sidebar.slider("Frame Duration (ms)", 20, 100, 50))
    pivot_location = st.sidebar.selectbox(
        "Pivot Attachment", ["Bottom (e.g., Ear)", "Top (e.g., Leg)"]
    )

    # 2. Preview with ROI bounding box
    preview_img = img.copy()
    cv2.rectangle(preview_img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Selection Preview")
        st.image(
            cv2.cvtColor(preview_img, cv2.COLOR_BGR2RGB),
            caption="Green box marks the active search region",
            use_container_width=True,
        )

    # 3. Generate Part Mask
    ear_mask = np.zeros((h, w), dtype=np.uint8)
    roi = img[ymin:ymax, xmin:xmax]

    if roi.size > 0:
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # Select pixels darker than off-white
        _, thresh = cv2.threshold(gray_roi, 240, 255, cv2.THRESH_BINARY_INV)
        ear_mask[ymin:ymax, xmin:xmax] = thresh

        kernel = np.ones((5, 5), np.uint8)
        ear_mask = cv2.morphologyEx(ear_mask, cv2.MORPH_CLOSE, kernel)

    y_indices, x_indices = np.where(ear_mask > 0)

    with col2:
        st.subheader("Animation Output")
        if len(y_indices) == 0:
            st.warning(
                "No foreground pixels detected inside the green selection box. Adjust the sliders."
            )
        else:
            # Determine pivot based on selection type
            if "Bottom" in pivot_location:
                raw_y = np.max(y_indices)
            else:
                raw_y = np.min(y_indices)

            # Explicitly cast to native Python floats to prevent OpenCV TypeError
            pivot_y = float(raw_y)
            pivot_x = float(np.mean(x_indices[y_indices == int(pivot_y)]))

            # Render button
            if st.button("Generate Animation", type="primary"):
                with st.spinner("Rendering frames..."):
                    # Extract limb with alpha channel
                    b, g, r = cv2.split(img)
                    part_rgba = cv2.merge([b, g, r, ear_mask])

                    # Inpaint behind the extracted part
                    clean_base = cv2.inpaint(
                        img, ear_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA
                    )

                    # Build animation frames
                    frames = []
                    angle_steps = np.sin(np.linspace(0, 2 * np.pi, 20)) * swing_angle

                    for angle in angle_steps:
                        # Ensure both coordinates and angle are native Python floats
                        rot_mat = cv2.getRotationMatrix2D(
                            (pivot_x, pivot_y), float(angle), 1.0
                        )
                        rotated_part = cv2.warpAffine(
                            part_rgba,
                            rot_mat,
                            (w, h),
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT,
                        )

                        # Composite over clean base image
                        frame = clean_base.copy()
                        alpha = rotated_part[:, :, 3] / 255.0

                        for c in range(3):
                            frame[:, :, c] = (1.0 - alpha) * frame[
                                :, :, c
                            ] + alpha * rotated_part[:, :, c]

                        frames.append(
                            Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        )

                    # Export GIF buffer
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

                    st.image(gif_bytes, caption="Generated Animation")
                    st.download_button(
                        label="Download GIF",
                        data=gif_bytes,
                        file_name="animated_drawing.gif",
                        mime="image/gif",
                    )
