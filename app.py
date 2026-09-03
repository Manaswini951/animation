import io
import cv2
import numpy as np
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Attached 2D Limb Animator", layout="wide")
st.title("Seamless 2D Limb & Ear Animator (No Detaching / No AI)")

uploaded_file = st.file_uploader(
    "Upload your hand-drawn character", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # 1. Load image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    h, w = img.shape[:2]

    # Sidebar: Color Detection & Elasticity Settings
    st.sidebar.header("1. Dark Pink Detection")
    h_min = st.sidebar.slider("Hue Min", 0, 180, 135)
    h_max = st.sidebar.slider("Hue Max", 0, 180, 175)
    s_min = st.sidebar.slider("Saturation Min", 0, 255, 60)
    v_min = st.sidebar.slider("Value/Brightness Min", 0, 255, 60)

    st.sidebar.header("2. Animation Physics")
    motion_type = st.sidebar.selectbox(
        "Movement Style",
        [
            "Ear / Limb Twitch (Anchored Base)",
            "Smile / Muzzle Stretch",
            "Breathing / Body Pulse (Root Anchored)",
        ],
    )
    elastic_stretch = float(st.sidebar.slider("Wiggle Reach (Pixels)", 3.0, 35.0, 12.0))
    stiffness = float(st.sidebar.slider("Joint Rigidity / Falloff", 1.0, 3.0, 1.6))
    frame_speed = int(st.sidebar.slider("Frame Delay (ms)", 20, 120, 45))

    # 2. Extract Marked Color (HSV)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_pink = np.array([h_min, s_min, v_min])
    upper_pink = np.array([h_max, 255, 255])
    raw_mask = cv2.inRange(hsv, lower_pink, upper_pink)

    # Clean gaps and noise
    kernel = np.ones((5, 5), np.uint8)
    clean_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, kernel)

    # 3. Detect Connected Components (Ears/Limbs)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        clean_mask, connectivity=8
    )

    parts_meta = []
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= 150:
            part_mask = (labels == i).astype(np.uint8)
            y_pts, x_pts = np.where(part_mask > 0)

            # Base anchor (pinned to head/body - lowest point)
            base_y = float(np.max(y_pts))
            base_x = float(np.mean(x_pts[y_pts == int(base_y)]))

            # Free tip (moves the most - highest point)
            tip_y = float(np.min(y_pts))
            tip_x = float(np.mean(x_pts[y_pts == int(tip_y)]))

            height_span = max(1.0, base_y - tip_y)

            parts_meta.append({
                "mask": part_mask,
                "base": (base_x, base_y),
                "tip": (tip_x, tip_y),
                "span": height_span,
            })

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Anchor Point Preview")
        preview = img.copy()
        for p in parts_meta:
            # Red dot = Fixed Anchor to body (0% motion)
            cv2.circle(
                preview,
                (int(p["base"][0]), int(p["base"][1])),
                6,
                (0, 0, 255),
                -1,
            )
            # Blue dot = Tip (100% motion)
            cv2.circle(
                preview,
                (int(p["tip"][0]), int(p["tip"][1])),
                5,
                (255, 0, 0),
                -1,
            )
            # Connecting line indicating tension vector
            cv2.line(
                preview,
                (int(p["base"][0]), int(p["base"][1])),
                (int(p["tip"][0]), int(p["tip"][1])),
                (0, 255, 0),
                2,
            )

        st.image(
            cv2.cvtColor(preview, cv2.COLOR_BGR2RGB),
            caption="Red Dot = Body Connection (Stationary) | Blue Dot = Moving Tip",
            use_container_width=True,
        )

    with col2:
        st.subheader("Seamless Animation")
        if len(parts_meta) == 0:
            st.warning("No marked color zones found. Adjust Hue/Saturation sliders.")
        else:
            if st.button("Generate Seamless Motion", type="primary"):
                with st.spinner("Calculating elastic mesh deformations..."):
                    # Pre-calculate base pixel coordinates
                    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
                    grid_x = grid_x.astype(np.float32)
                    grid_y = grid_y.astype(np.float32)

                    frames = []
                    steps = np.sin(np.linspace(0, 2 * np.pi, 24))

                    for step in steps:
                        # Copy coordinate maps
                        map_x = grid_x.copy()
                        map_y = grid_y.copy()

                        for idx, p in enumerate(parts_meta):
                            p_mask = p["mask"] == 1
                            base_x, base_y = p["base"]
                            span = p["span"]

                            # Weight = 0 at base (pinned), weight = 1 at tip
                            # Clamped between 0 and 1
                            vert_dist = np.clip((base_y - grid_y) / span, 0.0, 1.0)
                            weight = np.power(vert_dist, stiffness)

                            # Direction alternates for left vs right ears
                            side_multiplier = 1.0 if idx % 2 == 0 else -1.0

                            if "Twitch" in motion_type:
                                # Smooth horizontal sway, pinned at base
                                dx = step * elastic_stretch * side_multiplier * weight
                                map_x[p_mask] -= dx[p_mask]

                            elif "Smile" in motion_type:
                                # Quadratic parabolic arching
                                dx_center = (grid_x - base_x) / 40.0
                                dy = step * elastic_stretch * (dx_center ** 2) * weight
                                map_y[p_mask] -= dy[p_mask]

                            elif "Pulse" in motion_type:
                                # Vertical squash & stretch anchored to root
                                dy = step * elastic_stretch * weight
                                map_y[p_mask] -= dy[p_mask]

                        # Warp entire image smoothly without separating seams
                        deformed_frame = cv2.remap(
                            img,
                            map_x,
                            map_y,
                            interpolation=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REFLECT,
                        )

                        frames.append(
                            Image.fromarray(cv2.cvtColor(deformed_frame, cv2.COLOR_BGR2RGB))
                        )

                    # Export GIF
                    buf = io.BytesIO()
                    frames[0].save(
                        buf,
                        format="GIF",
                        save_all=True,
                        append_images=frames[1:],
                        duration=frame_speed,
                        loop=0,
                    )
                    gif_data = buf.getvalue()

                    st.image(gif_data, caption="Seamlessly Anchored Animation")
                    st.download_button(
                        "Download GIF", gif_data, "seamless_animation.gif", "image/gif"
                    )
