import pyrealsense2 as rs
import numpy as np
import cv2

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

pipeline.start(config)

align_to = rs.stream.color
align = rs.align(align_to)

temporal = rs.temporal_filter()
spatial = rs.spatial_filter()

kernel = np.ones((3, 3), np.uint8)

edge_color_bgr = [255, 255, 0]

try:
    while True:
        frames = pipeline.wait_for_frames()

        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        depth_frame = spatial.process(depth_frame)
        depth_frame = temporal.process(depth_frame)

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        depth_float = depth_image.astype(np.float32)

        blur_depth = cv2.GaussianBlur(depth_float, (35, 35), 4.0)

        laplacian_depth = cv2.Laplacian(blur_depth, cv2.CV_32F)

        min_log = cv2.erode(laplacian_depth, kernel)
        max_log = cv2.dilate(laplacian_depth, kernel)

        zero_cross = np.logical_and(min_log < 0, max_log > 0)

        log_diff = max_log - min_log
        threshold = 30.0
        strong_edges_mask = log_diff > threshold

        raw_edges_mask = np.logical_and(zero_cross, strong_edges_mask)
        final_edges_mask = raw_edges_mask > 0

        color_with_edges = color_image.copy()
        color_with_edges[final_edges_mask] = edge_color_bgr

        #depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        #cv2.imshow('RealSense Overlay', depth_colormap)
        cv2.imshow('RealSense Overlay', color_with_edges)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()