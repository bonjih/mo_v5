import numpy as np
import cv2
#
# def main():
#     cap = cv2.VideoCapture("./output/crusher_bin_bridge_4.mkv")
#     cap.set(cv2.CAP_PROP_POS_MSEC, 800 * 1.0e3)
#     polygon = []
#     points = []
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             print('EOF')
#             break
#
#         frame = cv2.polylines(frame, polygon, False, (255, 0, 0), thickness=5)
#
#         cv2.imshow('Frame', frame)
#
#         key = cv2.waitKey(25)
#         if key == ord('q'):
#             break
#         elif key == ord('p'):
#             polygon = [np.int32(points)]
#             points = []
#
#         cv2.setMouseCallback('Frame', left_click_detect, points)
#
#     cap.release()  # release video file
#     cv2.destroyAllWindows()  # close all openCV windows
#
#
# def left_click_detect(event, x, y, flags, points):
#     if event == cv2.EVENT_LBUTTONDOWN:
#         print(f"\t  {x}, {y}")
#         points.append([x, y])
#         #print(points)


# Function to draw polygon on the video
def draw_polygon(event, x, y, flags, param):
    global points, paused
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        if len(points) > 1:
            cv2.line(frame, points[-2], points[-1], (0, 255, 0), 2)
        cv2.imshow('Video', frame)
        paused = True
    print(points)



# Open video capture
cap = cv2.VideoCapture('./data/crusher_bin_bridge2.mkv')
cap.set(cv2.CAP_PROP_POS_MSEC, 800 * 1.0e3)

# Create a window and set the mouse callback function
cv2.namedWindow('Video')
cv2.setMouseCallback('Video', draw_polygon)

points = []
paused = False
while cap.isOpened():
    if not paused:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow('Video', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('p'):
        paused = not paused

cap.release()
cv2.destroyAllWindows()