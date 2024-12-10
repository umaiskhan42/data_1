from argparse import ArgumentParser
import tensorflow as tf
import numpy as np
import cv2
from pathlib import Path
from datetime import timedelta
import pandas as pd

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

# Define key codes
RAIL_KEY = ord('r')       # R key for rail (straight)
CROSS_COURT_KEY = ord('c') # C key for cross court
BOAST_KEY = ord('b')       # B key for boast
SERVE_KEY = ord('s')       # S key for serve
ESC_KEY = 27               # Escape key for exit
SPACE_KEY = 32             # Space key to pause/resume
FORWARD_KEY = ord('d')     # Right arrow key to move forward
BACKWARD_KEY = ord('a')    # Left arrow key to move backward

def format_timestamp(milliseconds):
    """Convert milliseconds to hh:mm:ss:ms format."""
    seconds = milliseconds / 1000
    timestamp = str(timedelta(seconds=seconds))
    # Ensure timestamp format is hh:mm:ss.ms (3 decimal places for ms)
    return timestamp if '.' in timestamp else timestamp + ".000"


if __name__ == "__main__":
    parser = ArgumentParser(description="Annotation Script")
    parser.add_argument("video")
    parser.add_argument("--debug",action="store_const",const=True,default=False,help="Show sub frame (RoI)",)
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    assert cap.isOpened()

        # Check if video opened successfully
    if not cap.isOpened():
        print("Error opening video stream or file")
        exit()

    #Extract the video filename
    video_name = Path(args.video).name  

    ret, frame = cap.read()
    assert ret, "Error reading the first frame."
    frame= cv2.resize(frame, (640,640))

    human_pose_extractor = HumanPoseExtractor(frame.shape)

    df = pd.DataFrame(columns=["Shot", "FrameId", "Timestamp", "VideoName"])
    FRAME_ID = 0
    your_list = []
    paused = False  # Video paused flag

    while cap.isOpened():
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break

            FRAME_ID += 1

            frame= cv2.resize(frame,(640,640))
            human_pose_extractor.extract(frame)

        # dont draw non-significant points/edges by setting probability to 0
            human_pose_extractor.discard(["left_eye", "right_eye", "left_ear", "right_ear"])

            # Get the current timestamp in milliseconds and format it
            timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            timestamp_formatted = format_timestamp(timestamp_ms)
        
            # Check for key presses
            k = cv2.waitKey(30) & 0xFF
            print(f"Detected key code: {k}")

            # Handle key presses for shot annotation
            if k == RAIL_KEY:
                your_list.append({"Shot": "rail", "FrameId": FRAME_ID, "Timestamp": timestamp_formatted, "VideoName": video_name})
                print(f"Added rail at frame {FRAME_ID}, timestamp {timestamp_formatted}")
            elif k == CROSS_COURT_KEY:
                your_list.append({"Shot": "cross_court", "FrameId": FRAME_ID, "Timestamp": timestamp_formatted, "VideoName": video_name})
                print(f"Added cross court at frame {FRAME_ID}, timestamp {timestamp_formatted}")
            elif k == BOAST_KEY:
                your_list.append({"Shot": "boast", "FrameId": FRAME_ID, "Timestamp": timestamp_formatted, "VideoName": video_name})
                print(f"Added boast at frame {FRAME_ID}, timestamp {timestamp_formatted}")
            elif k == SERVE_KEY:
                your_list.append({"Shot": "serve", "FrameId": FRAME_ID, "Timestamp": timestamp_formatted, "VideoName": video_name})
                print(f"Added serve at frame {FRAME_ID}, timestamp {timestamp_formatted}")

            # Pause/Resume with SPACE
            if k == SPACE_KEY:
                paused = not paused  # Toggle pause state
                print("Paused" if paused else "Resumed")


            # Move forward by 10 frames
            elif k == FORWARD_KEY:
                FRAME_ID += 60
                cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_ID)
                print(f"Moved forward to frame {FRAME_ID}")

            # Move backward by 10 frames
            elif k == BACKWARD_KEY:
                FRAME_ID = max(0, FRAME_ID - 60)  # Prevent negative frame ID
                cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_ID)
                print(f"Moved backward to frame {FRAME_ID}")

            # Exit on ESC key
            if k == ESC_KEY:
                print("Exiting")
                break

            FRAME_ID += 1

        else:
            # If paused, wait for user input to resume or exit
            k = cv2.waitKey(0) & 0xFF
            if k == SPACE_KEY:  # Resume if SPACE is pressed
                paused = False
                print("Resumed")
            elif k == ESC_KEY:  # Exit if ESC is pressed
                print("Exiting")
                break

        # Extract subframe (roi) and display results
        if args.debug:
            subframe = human_pose_extractor.draw_results_subframe()
            cv2.imshow("Subframe", subframe)

        # Display results on original frame
        frame= cv2.resize(frame, (640,640))
        human_pose_extractor.draw_results_frame(frame)
        cv2.imshow("Frame", frame)
        human_pose_extractor.roi.update(human_pose_extractor.keypoints_pixels_frame)

    # Save annotations if any
    if your_list:
        out_file = f"annotation_{Path(args.video).stem}.csv"
        df = pd.DataFrame.from_records(your_list)
        df.to_csv(out_file, index=False)
        print(f"Annotation file was written to {out_file}")
    else:
        print("No annotations were made.")

    cap.release()
    cv2.destroyAllWindows()


