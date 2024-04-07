from collections import deque

import numpy as np
import cv2

from compute_psd import compute_psd
from dust_detect import detect_blur_fft
import global_params_variables
from vid_lables import dust, timestamp, display, draw_roi_poly

# setup
params = global_params_variables.ParamsDict()

motion_offset = params.get_value('motion_offset')

psd_deque = deque(maxlen=motion_offset)

motion_intensity_history = []  # List to store past motion intensity values
window_size = 20  # Size of the moving average window


def pre_process(prev_frame, roi_comp):
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(prev_gray)

    for roi_key in roi_comp.rois:
        roi = roi_comp.rois[roi_key]
        cv2.fillPoly(mask, [roi.get_polygon_points()], 255)

    return prev_gray, mask


def create_roi_mask(frame_shape, roi_points):
    mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [roi_points], 255)
    return mask


def update_motion_intensity_history(motion_intensity):
    # Append the new motion intensity value to the history
    motion_intensity_history.append(motion_intensity)
    # Maintain the history size by removing oldest values if necessary
    if len(motion_intensity_history) > window_size:
        motion_intensity_history.pop(0)


def compute_filtered_motion_intensity():
    # Calculate the moving average of motion intensity values
    if len(motion_intensity_history) > 0:
        return sum(motion_intensity_history) / len(motion_intensity_history)
    else:
        return 0  # Return 0 if history is empty


def process_frame_motion(ts, roi_comp, frame, prev_gray, mask, motion_detected, motion_frames, motion_start_frame):
    motion_intensities_per_roi = []
    avg_intensity_per_roi = []

    min_contour_area = 100
    psd_val = 0
    text_y = 50

    mean, dusty = detect_blur_fft(frame)
    # display on output video
    dust(frame, mean, dusty)
    timestamp(frame, ts)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray, prev_gray)
    diff_roi = cv2.bitwise_and(diff, diff, mask=mask)

    # Apply adaptive thresholding to improve robustness against varying lighting conditions
    diff_roi = cv2.adaptiveThreshold(diff_roi, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    _, thresh = cv2.threshold(diff_roi, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 10:
        total_area = sum(cv2.contourArea(contour) for contour in contours)
        if total_area > min_contour_area:
            if not motion_detected:
                motion_detected = True
                motion_start_frame = motion_frames

    if motion_detected and motion_frames - motion_start_frame > motion_offset:
        for contour in contours:
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            cv2.drawContours(frame, [approx], -1, (0, 255, 0), 1)

    for roi_key in roi_comp.rois:
        roi = roi_comp.rois[roi_key]
        roi_points = roi.get_polygon_points()
        roi_mask = create_roi_mask(frame.shape, roi_points)
        draw_roi_poly(frame, roi_points)
        # Calculate motion intensity within the ROI
        diff_roi_roi = cv2.bitwise_and(diff_roi, diff_roi, mask=roi_mask)
        flattened_data = np.concatenate(diff_roi_roi)
        normalized_data = (flattened_data - np.min(flattened_data)) / (
                np.max(flattened_data) - np.min(flattened_data))

        sum_features = np.std(normalized_data.flatten()) * 100

        motion_intensity = (np.mean(diff_roi_roi))
        motion_intensities_per_roi.append(motion_intensity)

        avg_intensity = np.median(np.array(motion_intensities_per_roi))
        avg_intensity_per_roi.append(avg_intensity)

        # Update motion intensity history and compute filtered motion intensity
        update_motion_intensity_history(motion_intensity)
        filtered_motion_intensity = compute_filtered_motion_intensity()

        if len(psd_deque) >= motion_offset:
            psd_val = compute_psd(list(psd_deque))

        psd_deque.append([ts, motion_intensity])

        if sum_features > 11 and psd_val < 6.5:
            bridge_text = f"{roi_key} Normal [{filtered_motion_intensity:.4f} {sum_features:.4f} {psd_val:.4f}]"
            is_bridge = 0
        elif -11 < psd_val < -7:
            bridge_text = f"{roi_key} Potential Bridge [{filtered_motion_intensity:.4f} {sum_features:.4f} {psd_val:.4f}"
            is_bridge = 0
        elif sum_features > 16 and psd_val < -10:
            bridge_text = f"{roi_key} Bridge [{filtered_motion_intensity:.4f} {sum_features:.4f} {psd_val:.4f}"
            is_bridge = 0
        else:
            bridge_text = f"{roi_key} No Bridge [{filtered_motion_intensity:.4f} {sum_features:.4f} {psd_val:.4f}"
            is_bridge = 0

        if not dusty:
            bridge_color = (0, 0, 255) if "Bridge" in bridge_text else (0, 255, 0)
            cv2.putText(frame, bridge_text, (10, text_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, bridge_color,
                        1)

            cv2.putText(frame, roi_key, (roi_points[1:2, 0:1][-1][-1], roi_points[1:2, 1:][-1][-1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1)

        text_y += 30

    motion_frames += 1
    prev_gray = gray.copy()

    return motion_detected, motion_frames, prev_gray
