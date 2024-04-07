import cv2

from motion_diff import pre_process, process_frame_motion


class VideoProcessor:
    def __init__(self, video_path, output_dir, roi_comp, is_watching=False, is_save_video=False, offset=0):
        self.video_path = video_path
        self.output_dir = output_dir
        self.roi_comp = roi_comp
        self.is_watching = is_watching
        self.is_save_video = is_save_video
        self.offset = offset

    def process_video(self):
        out = None
        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_POS_MSEC, self.offset * 1.0e3)

        if self.is_save_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(self.output_dir, fourcc, fps, (frame_width, frame_height))

        _, prev_frame = cap.read()

        prev_gray, mask = pre_process(prev_frame, self.roi_comp)

        motion_detected = True
        motion_frames = 0
        motion_start_frame = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if prev_frame is None:
                prev_frame = frame

            ts = cap.get(cv2.CAP_PROP_POS_MSEC)
            motion_detected, motion_frames, prev_gray = process_frame_motion(ts, self.roi_comp, frame,
                                                                             prev_gray, mask,
                                                                             motion_detected,
                                                                             motion_frames,
                                                                             motion_start_frame)

            if self.is_watching:
                cv2.imshow('Filtered Frame ', frame)

            if self.is_save_video:
                out.write(frame)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        cap.release()
        if self.is_save_video:
            out.release()
        cv2.destroyAllWindows()

    def start(self):
        self.process_video()
