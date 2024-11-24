from utils import (read_video,save_video)
from trackers import PlayerTracker,BallTracker


def main():
    # read video
    input_video_path = "new\input_videos\input_video.mp4"
    video_frames = read_video(input_video_path)

    # detect players
    player_tracker=PlayerTracker(model_path='new/yolov8x')
    ball_tracker=BallTracker(model_path='new/models/yolo8_last.pt')

    player_detections = player_tracker.detect_frames(video_frames,
                                                     read_from_stub=True,
                                                     stub_path="new/tracker_stubs/player_detections.pkl"
                                                     )
    
    ball_detections = ball_tracker.detect_frames(video_frames,
                                                     read_from_stub=True,
                                                     stub_path="new/tracker_stubs/ball_detections.pkl"
                                                     )
    
    ## Draw Player Bounding Boxes
    output_video_frames= player_tracker.draw_bboxes(video_frames, player_detections)
    output_video_frames= ball_tracker.draw_bboxes(output_video_frames, ball_detections)


    save_video(output_video_frames,"new/output_videos/output_video.avi")

    
if __name__ == "__main__":
    main()