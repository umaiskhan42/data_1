import cv2
import pandas as pd
from pathlib import Path
from argparse import ArgumentParser
from typing import List, Dict
import tensorflow as tf
import numpy as np
# Constants for key presses
RAIL_KEY = ord("r")
CROSS_COURT_KEY = ord("c")
BOAST_KEY = ord("b")
SERVE_KEY = ord("s")
SPACE_KEY = 32  # Spacebar
ESC_KEY = 27    # Escape key
FORWARD_KEY = ord('d')     # Right arrow key to move forward
BACKWARD_KEY = ord('a')    # Left arrow key to move backward
REMOVE_KEY = ord("x")  # Key to remove annotation

# create a ROI bounding box around the player.
class RoI:

    def __init__(self, shape):
        self.frame_width = shape[1]
        self.frame_height = shape[0]
        self.width = self.frame_width
        self.height = self.frame_height
        self.center_x = shape[1] // 2
        self.center_y = shape[0] // 2
        self.valid = False

    def extract_subframe(self, frame):
        """Extract the RoI from the original frame"""
        subframe = frame.copy()
        return subframe[
            self.center_y - self.height // 2 : self.center_y + self.height // 2,
            self.center_x - self.width // 2 : self.center_x + self.width // 2,
        ]

    def transform_to_subframe_coordinates(self, keypoints_from_tf):
        """Key points from tensorflow come as float number betwen 0 and 1,
        describing (x, y) coordinates in the image feeding the NN
        We transform them into sub frame pixel coordinates
        """
        return np.squeeze(
            np.multiply(keypoints_from_tf, [self.width, self.width, 1])
        ) - np.array([(self.width - self.height) // 2, 0, 0])

    def transform_to_frame_coordinates(self, keypoints_from_tf):
        """Key points from tensorflow come as float number betwen 0 and 1,
        describing (x, y) coordinates in the image feeding the NN
        We transform them into frame pixel coordinates
        """
        keypoints_pixels_subframe = self.transform_to_subframe_coordinates(
            keypoints_from_tf
        )
        keypoints_pixels_frame = keypoints_pixels_subframe.copy()
        keypoints_pixels_frame[:, 0] += self.center_y - self.height // 2
        keypoints_pixels_frame[:, 1] += self.center_x - self.width // 2

        return keypoints_pixels_frame

    def update(self, keypoints_pixels):
        """Update RoI with new keypoints"""
        min_x = int(min(keypoints_pixels[:, 1]))
        min_y = int(min(keypoints_pixels[:, 0]))
        max_x = int(max(keypoints_pixels[:, 1]))
        max_y = int(max(keypoints_pixels[:, 0]))

        self.center_x = (min_x + max_x) // 2
        self.center_y = (min_y + max_y) // 2

        prob_mean = np.mean(keypoints_pixels[keypoints_pixels[:, 2] != 0][:, 2])
        if self.width != self.frame_width and prob_mean < 0.2:
            print(
                f"Lost player track --> reset ROI because prob is too low = {prob_mean}"
            )
            self.reset()
            return

        # keep next dimensions always a bit larger
        self.width = int((max_x - min_x) * 1.3)
        self.height = int((max_y - min_y) * 1.3)

        if self.height < 150 or self.width < 10:
            print(
                f"Lost player track --> reset ROI because height = {self.height} "
                f"and width = {self.width}"
            )
            self.reset()
            return

        self.width = max(self.width, self.height)
        self.height = max(self.width, self.height)

        if self.center_x + self.width // 2 >= self.frame_width:
            self.center_x = self.frame_width - self.width // 2 - 1

        if 0 > self.center_x - self.width // 2:
            self.center_x = self.width // 2 + 1

        if self.center_y + self.height // 2 >= self.frame_height:
            self.center_y = self.frame_height - self.height // 2 - 2

        if 0 > self.center_y - self.height // 2:
            self.center_y = self.height // 2 + 1

        # Reset if Out of Bound
        if self.center_x + self.width // 2 >= self.frame_width:
            self.reset()
            return

        # Reset if Out of Bound
        if self.center_y + self.height // 2 >= self.frame_height:
            self.reset()
            return

        assert 0 <= self.center_x - self.width // 2
        assert self.center_x + self.width // 2 < self.frame_width
        assert 0 <= self.center_y - self.height // 2
        assert self.center_y + self.height // 2 < self.frame_height

        # Set valid to True
        self.valid = True

    def reset(self):
        """
        Reset the RoI with width/height corresponding to the whole image
        """
        self.width = self.frame_width
        self.height = self.frame_height
        self.center_x = self.frame_width // 2
        self.center_y = self.frame_height // 2

        self.valid = False

    def draw_shot(self, frame, shot):
        """Draw shot name in orange around bounding box"""
        cv2.putText(
            frame,
            shot,
            (self.center_x - 50, self.center_y - self.height // 2 - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.8,
            color=(128, 255, 255),
            thickness=2,
        )
    def draw_label(self, frame, label):
        """Draw player label around the bounding box."""
        cv2.putText(
            frame,
            label,
            (self.center_x - 50, self.center_y - self.height // 2 - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.8,
            color=(255, 0, 0),
            thickness=2,
        )

class HumanPoseExtractor:
    """
    Defines mapping between movenet key points and human readable body points
    with realistic edges to be drawn"""

    EDGES = {
        (0, 1): "m",
        (0, 2): "c",
        (1, 3): "m",
        (2, 4): "c",
        (0, 5): "m",
        (0, 6): "c",
        (5, 7): "m",
        (7, 9): "m",
        (6, 8): "c",
        (8, 10): "c",
        (5, 6): "y",
        (5, 11): "m",
        (6, 12): "c",
        (11, 12): "y",
        (11, 13): "m",
        (13, 15): "m",
        (12, 14): "c",
        (14, 16): "c",
    }

    COLORS = {"c": (255, 255, 0), "m": (255, 0, 255), "y": (0, 255, 255)}

    # Dictionary that maps from joint names to keypoint indices.
    KEYPOINT_DICT = {
        "nose": 0,
        "left_eye": 1,
        "right_eye": 2,
        "left_ear": 3,
        "right_ear": 4,
        "left_shoulder": 5,
        "right_shoulder": 6,
        "left_elbow": 7,
        "right_elbow": 8,
        "left_wrist": 9,
        "right_wrist": 10,
        "left_hip": 11,
        "right_hip": 12,
        "left_knee": 13,
        "right_knee": 14,
        "left_ankle": 15,
        "right_ankle": 16,
    }

    def __init__(self, shape):
        # Initialize the TFLite interpreter
        self.interpreter = tf.lite.Interpreter(model_path="models/4.tflite")
        self.interpreter.allocate_tensors()

        self.roi = RoI(shape)
   
    def extract(self, frame):
        """Run inference model on subframe"""
        # Reshape image
        subframe = self.roi.extract_subframe(frame)

        img = subframe.copy()
        img = tf.image.resize_with_pad(np.expand_dims(img, axis=0), 192, 192)
        input_image = tf.cast(img, dtype=tf.uint8)
        # input_image = tf.cast(img, dtype=tf.int32)

        # Setup input and output
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()

        # Make predictions
        self.interpreter.set_tensor(input_details[0]["index"], np.array(input_image))
        #self.interpreter.set_tensor(input_details[0]["index"], np.expand_dims(input_image, axis=0))

        self.interpreter.invoke()
        self.keypoints_with_scores = self.interpreter.get_tensor(
            output_details[0]["index"]
        )
        self.keypoints_pixels_frame = self.roi.transform_to_frame_coordinates(
            self.keypoints_with_scores
        )

    def discard(self, list_of_keypoints):
        """Discard some points like eyes or ears (useless for our application)"""
        for keypoint in list_of_keypoints:
            self.keypoints_with_scores[0, 0, self.KEYPOINT_DICT[keypoint], 2] = 0
            self.keypoints_pixels_frame[self.KEYPOINT_DICT[keypoint], 2] = 0

    def draw_results_subframe(self):
        """Draw key points and edges on subframe (roi)"""
        subframe = self.roi.extract_subframe(frame)
        keypoints_pixels_subframe = self.roi.transform_to_subframe_coordinates(
            self.keypoints_with_scores
        )

        # Rendering
        draw_edges(subframe, keypoints_pixels_subframe, self.EDGES, 0.2)
        draw_keypoints(subframe, keypoints_pixels_subframe, 0.2)

        return subframe

    def draw_results_frame(self, frame):
        """Draw key points and eges on frame"""
        if not self.roi.valid:
            return

        draw_edges(frame, self.keypoints_pixels_frame, self.EDGES, 0.01)
        draw_keypoints(frame, self.keypoints_pixels_frame, 0.01)
        draw_roi(self.roi, frame)
        #draw_results(self,frame)

def draw_keypoints(frame, shaped, confidence_threshold):
    """Draw key points with green dots"""
    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 4, (0, 255, 0), -1)

def draw_edges(frame, shaped, edges, confidence_threshold):
    """Draw edges with cyan for the right side, magenta for the left side, rest in yellow"""
    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]
        if (c1 > confidence_threshold) & (c2 > confidence_threshold):
            cv2.line(
                frame,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                color=HumanPoseExtractor.COLORS[color],
                thickness=2,
            )
def draw_roi(roi, frame):
    """Draw RoI with a yellow square"""
    cv2.line(
        frame,
        (roi.center_x - roi.width // 2, roi.center_y - roi.height // 2),
        (roi.center_x - roi.width // 2, roi.center_y + roi.height // 2),
        (0, 255, 255),
        3,
    )
    cv2.line(
        frame,
        (roi.center_x + roi.width // 2, roi.center_y + roi.height // 2),
        (roi.center_x - roi.width // 2, roi.center_y + roi.height // 2),
        (0, 255, 255),
        3,
    )
    cv2.line(
        frame,
        (roi.center_x + roi.width // 2, roi.center_y + roi.height // 2),
        (roi.center_x + roi.width // 2, roi.center_y - roi.height // 2),
        (0, 255, 255),
        3,
    )
    cv2.line(
        frame,
        (roi.center_x - roi.width // 2, roi.center_y - roi.height // 2),
        (roi.center_x + roi.width // 2, roi.center_y - roi.height // 2),
        (0, 255, 255),
        3,
    )

def format_timestamp(timestamp_ms):
    seconds = timestamp_ms / 1000
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours):02}:{int(minutes):02}:{seconds:06.3f}"

if __name__ == "__main__":
    parser = ArgumentParser(description="Video Annotation Script")
    parser.add_argument("video", help="Path to the video file.")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode for additional visuals.")
    parser.add_argument("--load_annotations", type=str, help="Path to a saved annotations CSV file.")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        exit()

    video_name = Path(args.video).stem
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame.")
        exit()

    frame = cv2.resize(frame, (640, 640))
    human_pose_extractor = HumanPoseExtractor(frame.shape)
    annotations_list: List[Dict] = []

    annotations = []
    if args.load_annotations:
        try:
            annotations_df = pd.read_csv(args.load_annotations)
            annotations = annotations_df.to_dict("records")
            print(f"Loaded {len(annotations)} annotations from {args.load_annotations}.")
        except Exception as e:
            print(f"Error loading annotations: {e}")
            exit()

    FRAME_ID = 0
    paused = False

    while cap.isOpened():
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break

            FRAME_ID += 1
            frame = cv2.resize(frame, (640, 640))
            human_pose_extractor.extract(frame)
            human_pose_extractor.discard(["left_eye", "right_eye", "left_ear", "right_ear"])

            for annotation in annotations:
                if annotation["FrameId"] == FRAME_ID:
                    human_pose_extractor.roi.draw_shot(frame, annotation["Shot"])
                    paused = True  # Pause video when an annotation is displayed
                    print(f"Paused at frame {FRAME_ID} with annotation: {annotation['Shot']}")
                    break

            human_pose_extractor.draw_results_frame(frame)

        frame_number_text = f"Frame: {FRAME_ID}"
        text_size, _ = cv2.getTextSize(frame_number_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        text_x = frame.shape[1] - text_size[0] - 10
        text_y = 30
        cv2.putText(frame, frame_number_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow("Video Annotation", frame)
        key = cv2.waitKey(30) & 0xFF
        timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        timestamp_formatted = format_timestamp(timestamp_ms)

        if paused:
            if key == RAIL_KEY:
                annotations = [
                    {**annotation, "Shot": "rail"} if annotation["FrameId"] == FRAME_ID else annotation
                    for annotation in annotations
                ]
                print(f"Updated to rail at frame {FRAME_ID}, timestamp {timestamp_formatted}")
                paused = False
            elif key == CROSS_COURT_KEY:
                annotations = [
                    {**annotation, "Shot": "cross_court"} if annotation["FrameId"] == FRAME_ID else annotation
                    for annotation in annotations
                ]
                print(f"Updated to cross court at frame {FRAME_ID}, timestamp {timestamp_formatted}")
                paused = False
            elif key == BOAST_KEY:
                annotations = [
                    {**annotation, "Shot": "boast"} if annotation["FrameId"] == FRAME_ID else annotation
                    for annotation in annotations
                ]
                print(f"Updated to boast at frame {FRAME_ID}, timestamp {timestamp_formatted}")
                paused = False
            elif key == SERVE_KEY:
                annotations = [
                    {**annotation, "Shot": "serve"} if annotation["FrameId"] == FRAME_ID else annotation
                    for annotation in annotations
                ]
                print(f"Updated to serve at frame {FRAME_ID}, timestamp {timestamp_formatted}")
                paused = False
            elif key == REMOVE_KEY:
                annotations = [
                    annotation for annotation in annotations if annotation["FrameId"] != FRAME_ID
                ]
                print(f"Removed annotation at frame {FRAME_ID}, timestamp {timestamp_formatted}")
                paused = False
            elif key == SPACE_KEY:
                paused = False
                print("Resumed playback.")

            # Move forward by 10 frames
        if key == FORWARD_KEY:
                FRAME_ID += 60
                cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_ID)
                print(f"Moved forward to frame {FRAME_ID}")

            # Move backward by 10 frames
        if key == BACKWARD_KEY:
                FRAME_ID = max(0, FRAME_ID - 60)  # Prevent negative frame ID
                cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_ID)
                print(f"Moved backward to frame {FRAME_ID}")

        if key == ESC_KEY:
                break

    save_path = f"{video_name}_modified_annotations.csv"
    pd.DataFrame(annotations).to_csv(save_path, index=False)
    print(f"Modified annotations saved to {save_path}.")

    cap.release()
    cv2.destroyAllWindows()
