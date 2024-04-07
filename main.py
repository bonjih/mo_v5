import os

from RoiMultiClass import ComposeROI
from video_streamer import VideoProcessor
import global_params_variables


def main():
    params = global_params_variables.ParamsDict()

    roi_config = ComposeROI(params.get_all_items())
    video_path = roi_config.video_file

    if not os.path.exists(video_path) or not os.path.isfile(video_path):
        print("Input video path or file does not exist.")
        return

    output_path = params.get_value('output_video_path')

    if output_path is None:
        output_dir = params.get_value('output_dir')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "output_video.mkv")

    is_watching = params.get_value('is_watching')
    is_save_video = params.get_value('is_save_video')
    offset = params.get_value('offset')

    video_processor = VideoProcessor(video_path, output_path, roi_config,
                                     is_watching=is_watching,
                                     is_save_video=is_save_video,
                                     offset=offset)
    video_processor.process_video()


if __name__ == '__main__':
    main()
