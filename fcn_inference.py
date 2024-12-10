from argparse import ArgumentParser
import tensorflow as tf
import cv2
import numpy as np
import pandas as pd
from tensorflow import keras
from annotation_visualization import HumanPoseExtractor
from track_and_classify_with_rnn import GT, draw_probs

class ShotCounter:

    MIN_FRAMES_BETWEEN_SHOTS = 60

    BAR_WIDTH = 30
    BAR_HEIGHT = 170
    MARGIN_ABOVE_BAR = 30
    SPACE_BETWEEN_BARS = 55
    TEXT_ORIGIN_X = 1075
    BAR_ORIGIN_X = 1070

    def __init__(self):
        self.nb_history = 10 
        self.probs = np.zeros((self.nb_history, 4))

        self.nb_serves = 0
        self.nb_rallies = 0
        self.current_rally_shots = 0
        self.shots_per_rally = []

        self.last_shot = "neutral"
        self.frames_since_last_shot = self.MIN_FRAMES_BETWEEN_SHOTS

        self.results = []

    def update(self, probs, frame_id):
        """Update current state with new shots probabilities"""
        self.probs[0 : self.nb_history - 1, :] = self.probs[1:, :].copy()
        self.probs[-1, :] = probs

        self.frames_since_last_shot += 1

        means = np.mean(self.probs, axis=0)
        
        if means[0] > 0.5 or means[1] > 0.5:  # Merging forehand and backhand as "shot"
            if (
                self.last_shot == "neutral"
                and self.frames_since_last_shot > self.MIN_FRAMES_BETWEEN_SHOTS
            ):
                self.current_rally_shots += 1
                self.last_shot = "shot"
                self.frames_since_last_shot = 0
                self.results.append({"FrameID": frame_id, "Shot": self.last_shot})

        elif means[2] > 0.5:
            self.last_shot = "neutral"

        elif means[3] > 0.5:  # Serve detection
            if (
                self.last_shot == "neutral"
                and self.frames_since_last_shot > self.MIN_FRAMES_BETWEEN_SHOTS
            ):
                if self.current_rally_shots > 0:
                    self.nb_rallies += 1  # Count completed rally
                    self.shots_per_rally.append(self.current_rally_shots)  # Record shots in the rally
                self.nb_serves += 1
                self.current_rally_shots = 0  # Reset rally shot count for next rally
                self.last_shot = "serve"
                self.frames_since_last_shot = 0
                self.results.append({"FrameID": frame_id, "Shot": self.last_shot})

    def display(self, frame):
        """Display shot and rally counts, and shots per rally"""
        cv2.putText(
            frame,
            f"Rallies = {self.nb_rallies}",
            (20, frame.shape[0] - 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(0, 255, 0) if self.last_shot == "serve" and self.frames_since_last_shot < 30 else (0, 0, 255),
            thickness=2,
        )
        cv2.putText(
            frame,
            f"Shots in current rally = {self.current_rally_shots}",
            (20, frame.shape[0] - 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(0, 255, 0) if self.last_shot == "shot" and self.frames_since_last_shot < 30 else (0, 0, 255),
            thickness=2,
        )
        cv2.putText(
            frame,
            f"Serves = {self.nb_serves}",
            (20, frame.shape[0] - 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(0, 255, 0) if self.last_shot == "serve" and self.frames_since_last_shot < 30 else (0, 0, 255),
            thickness=2,
        )
        # cv2.putText(
        #     frame,        
        #     f"Shots per rally: {self.shots_per_rally}",
        #     (20, frame.shape[0] - 60),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     fontScale=0.6,
        #     color=(255, 255, 255),
        #     thickness=1,
        # )

def compute_recall_precision(gt, shots):
    """Compute recall and precision metrics"""
    gt_numpy = gt.to_numpy()
    nb_match = 0
    nb_misses = 0
    nb_fp = 0
    fp_backhands = 0
    fp_forehands = 0
    fp_serves = 0
    for gt_shot in gt_numpy:
        found_match = False
        for shot in shots:
            if shot["Shot"] == gt_shot[0]:
                if abs(shot["FrameID"] - gt_shot[1]) <= 30:
                    found_match = True
                    break
        if found_match:
            nb_match += 1
        else:
            nb_misses += 1

    for shot in shots:
        found_match = False
        for gt_shot in gt_numpy:
            if shot["Shot"] == gt_shot[0]:
                if abs(shot["FrameID"] - gt_shot[1]) <= 30:
                    found_match = True
                    break
        if not found_match:
            nb_fp += 1
            if shot["Shot"] == "rail":
                fp_backhands += 1
            elif shot["Shot"] == "boast":
                fp_forehands += 1
            elif shot["Shot"] == "serve":
                fp_serves += 1

    precision = nb_match / (nb_match + nb_fp)
    recall = nb_match / (nb_match + nb_misses)

    print(f"Recall {recall*100:.1f}%")
    print(f"Precision {precision*100:.1f}%")

    print(
        f"FP: backhands = {fp_backhands}, forehands = {fp_forehands}, serves = {fp_serves}"
    )

if __name__ == "__main__":
    parser = ArgumentParser(description="Track tennis player and display shot probabilities")
    parser.add_argument("video")
    parser.add_argument("model")
    parser.add_argument("--evaluate", help="Path to annotation file")
    parser.add_argument("-f", type=int, help="Forward to")
    args = parser.parse_args()

    shot_counter = ShotCounter()

    if args.evaluate is not None:
        gt = GT(args.evaluate)

    m1 = keras.models.load_model(args.model)

    cap = cv2.VideoCapture(args.video)
    assert cap.isOpened()

    ret, frame = cap.read()

    # Create VideoWriter to save the processed video
    output_video = cv2.VideoWriter(
        'output_video1111.mp4', 
        cv2.VideoWriter_fourcc(*'mp4v'),  # Codec for MP4
        cap.get(cv2.CAP_PROP_FPS),  # Use the same frame rate as the input video
        (frame.shape[1], frame.shape[0])  # Use the same frame size as the input video
    )

    human_pose_extractor = HumanPoseExtractor(frame.shape)

    FRAME_ID = 0

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        FRAME_ID += 1

        if args.f is not None and FRAME_ID < args.f:
            continue

        assert frame is not None

        human_pose_extractor.extract(frame)
        human_pose_extractor.discard(["left_eye", "right_eye", "left_ear", "right_ear"])

        features = human_pose_extractor.keypoints_with_scores.reshape(17, 3)
        features = features[features[:, 2] > 0][:, 0:2].reshape(1, 13 * 2)

        probs = (
            m1.__call__(features)[0] if human_pose_extractor.roi.valid else np.zeros(4)
        )

        shot_counter.update(probs, FRAME_ID)
        shot_counter.display(frame)

        if args.evaluate is not None:
            gt.display(frame, FRAME_ID)

        human_pose_extractor.draw_results_frame(frame)

        if (
            shot_counter.frames_since_last_shot < 30
            and shot_counter.last_shot != "neutral"
        ):
            human_pose_extractor.roi.draw_shot(frame, shot_counter.last_shot)

        # Write the processed frame to the output video
        output_video.write(frame)

        human_pose_extractor.roi.update(human_pose_extractor.keypoints_pixels_frame)

    cap.release()
    output_video.release()  # Release the VideoWriter
    cv2.destroyAllWindows()

    print(shot_counter.results)
    print("Shots per rally:", shot_counter.shots_per_rally)

    if args.evaluate is not None:
        compute_recall_precision(gt.shots, shot_counter.results)
