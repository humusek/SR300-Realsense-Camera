import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO

model_path = r"D:\Studia\cybAIR\Ambition\rocks_detection\Detection\runs\detect\depthai_model\yolo_rocks-9\weights\best.pt"
model = YOLO(model_path)

pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

profile = pipeline.start(config)

align_to = rs.stream.color
align = rs.align(align_to)

color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
intrinsics = color_profile.get_intrinsics()
fx = intrinsics.fx
fy = intrinsics.fy

depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        results = model.predict(source=color_image, conf=0.5, stream=False, verbose=False)
        annotated_frame = results[0].plot()

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            x1_safe = max(0, x1)
            y1_safe = max(0, y1)
            x2_safe = min(color_image.shape[1], x2)
            y2_safe = min(color_image.shape[0], y2)

            roi_depth = depth_image[y1_safe:y2_safe, x1_safe:x2_safe]
            valid_depths = roi_depth[roi_depth > 0]

            if len(valid_depths) > 0:
                z_raw = np.median(valid_depths)
                z_m = z_raw * depth_scale
                z_mm = z_m * 1000.0

                w_px = x2 - x1
                h_px = y2 - y1

                width_cm = ((w_px * z_mm) / fx) / 10.0
                height_cm = ((h_px * z_mm) / fy) / 10.0
                z_cm = z_mm / 10.0

                text = f"W:{width_cm:.1f}cm H:{height_cm:.1f}cm Z:{z_cm:.1f}cm"
                cv2.putText(annotated_frame, text, (x1, y2 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        cv2.imshow('SR300 RGB - Detekcja', annotated_frame)
        cv2.imshow('SR300 Glebia', depth_colormap)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    pipeline.stop()
cv2.destroyAllWindows()