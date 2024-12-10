# import time
# from argparse import ArgumentParser
# import tensorflow as tf
# import numpy as np
# import cv2
# from tensorflow import keras
# from annotation_visualization import HumanPoseExtractor

# class ShotCounter:
#     MIN_FRAMES_BETWEEN_SHOTS = 60

#     def __init__(self):
#         self.nb_history = 30
#         self.probs = np.zeros(5)

#         self.nb_cross_courts = 0
#         self.nb_boasts = 0
#         self.nb_rails = 0
#         self.nb_serves = 0
#         self.nb_no_shots = 0

#         self.last_shot = "neutral"
#         self.frames_since_last_shot = self.MIN_FRAMES_BETWEEN_SHOTS

#         self.results = []

#     def update(self, probs, frame_id):
#         if len(probs) == 5:
#             self.probs = probs

#         if probs[0] > 0.98 and self.frames_since_last_shot > self.MIN_FRAMES_BETWEEN_SHOTS:
#             self.nb_cross_courts += 1
#             self.last_shot = "cross_court"
#             self.frames_since_last_shot = 0
#             self.results.append({"FrameID": frame_id, "Shot": self.last_shot})
#         elif probs[1] > 0.98 and self.frames_since_last_shot > self.MIN_FRAMES_BETWEEN_SHOTS:
#             self.nb_boasts += 1
#             self.last_shot = "boast"
#             self.frames_since_last_shot = 0
#             self.results.append({"FrameID": frame_id, "Shot": self.last_shot})
#         elif probs[2] > 0.98 and self.frames_since_last_shot > self.MIN_FRAMES_BETWEEN_SHOTS:
#             self.nb_rails += 1
#             self.last_shot = "rail"
#             self.frames_since_last_shot = 0
#             self.results.append({"FrameID": frame_id, "Shot": self.last_shot})
#         elif probs[3] > 0.98 and self.frames_since_last_shot > self.MIN_FRAMES_BETWEEN_SHOTS:
#             self.nb_serves += 1
#             self.last_shot = "serve"
#             self.frames_since_last_shot = 0
#             self.results.append({"FrameID": frame_id, "Shot": self.last_shot})
#         elif probs[4] > 0.98 and self.frames_since_last_shot > self.MIN_FRAMES_BETWEEN_SHOTS:
#             self.nb_no_shots += 1
#             self.last_shot = "no_shot"
#             self.frames_since_last_shot = 0
#             self.results.append({"FrameID": frame_id, "Shot": self.last_shot})
#         self.frames_since_last_shot += 1

# # 'no_shot', 'boast', 'rail', 'cross_court', 'serve'
#     def display(self, frame):
#         cv2.putText(frame, f"Cross Court = {self.nb_cross_courts}", (20, frame.shape[0] - 140),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if (self.last_shot == "cross_court" and self.frames_since_last_shot < 30) else (0, 0, 255), 2)
#         cv2.putText(frame, f"Boast = {self.nb_boasts}", (20, frame.shape[0] - 110),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if (self.last_shot == "boast" and self.frames_since_last_shot < 30) else (0, 0, 255), 2)
#         cv2.putText(frame, f"Rail = {self.nb_rails}", (20, frame.shape[0] - 80),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if (self.last_shot == "rail" and self.frames_since_last_shot < 30) else (0, 0, 255), 2)
#         cv2.putText(frame, f"Serve = {self.nb_serves}", (20, frame.shape[0] - 50),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if (self.last_shot == "serve" and self.frames_since_last_shot < 30) else (0, 0, 255), 2)
#         cv2.putText(frame, f"No Shot = {self.nb_no_shots}", (20, frame.shape[0] - 20),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if (self.last_shot == "no_shot" and self.frames_since_last_shot < 30) else (0, 0, 255), 2)

# def draw_frame_id(frame, frame_id):
#     cv2.putText(frame, f"Frame {frame_id}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

# if __name__ == "__main__":
#     parser = ArgumentParser(description="Track tennis player and display shot probabilities")
#     parser.add_argument("video")
#     parser.add_argument("model")
#     parser.add_argument("--output", default="output.mp4", help="Output video filename")
#     args = parser.parse_args()

#     shot_counter = ShotCounter()

#     m1 = keras.models.load_model(args.model)

#     cap = cv2.VideoCapture(args.video)
#     assert cap.isOpened()

#     ret, frame = cap.read()
#     # assert frame is not None
#         # Define video writer
#     frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = int(cap.get(cv2.CAP_PROP_FPS))
#     out = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

#     human_pose_extractor = HumanPoseExtractor(frame.shape)

#     NB_IMAGES = 30
#     FRAME_ID = 0
#     features_pool = []
#     prev_time = time.time()

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         FRAME_ID += 1

#         human_pose_extractor.extract(frame)
#         human_pose_extractor.discard(["left_eye", "right_eye", "left_ear", "right_ear"])
#         features = human_pose_extractor.keypoints_with_scores.reshape(17, 3)
#         features = features[features[:, 2] > 0][:, 0:2].reshape(1, 13 * 2)
#         features_pool.append(features)

#         if len(features_pool) == NB_IMAGES:
#             features_seq = np.array(features_pool).reshape(1, NB_IMAGES, 26)
#             assert features_seq.shape == (1, 30, 26)
#             probs = (
#             m1.__call__(features_seq)[0] if human_pose_extractor.roi.valid else np.zeros(5)
#             )
#             #probs = m1.__call__(features_seq)[0]
#             shot_counter.update(probs, FRAME_ID)
#             features_pool = features_pool[1:]
#         #shot_counter.display(frame)
#         shot_counter.display(frame)

#         draw_frame_id(frame, FRAME_ID)

#         human_pose_extractor.draw_results_frame(frame)

#         human_pose_extractor.roi.update(human_pose_extractor.keypoints_pixels_frame)
#         # Write the processed frame to the output video
#         out.write(frame)

#     cap.release()
#     out.release()
#     cv2.destroyAllWindows()

###############
# import time
# from argparse import ArgumentParser
# import tensorflow as tf
# import numpy as np
# import cv2
# from tensorflow import keras
# from annotation_visualization import HumanPoseExtractor

# class ShotCounter:
#     MIN_FRAMES_BETWEEN_SHOTS = 60

#     def __init__(self):
#         self.nb_history = 30
#         self.probs = np.zeros(5)

#         self.nb_cross_courts = 0
#         self.nb_boasts = 0
#         self.nb_rails = 0
#         self.nb_serves = 0
#         self.nb_no_shots = 0

#         self.last_shot = "neutral"
#         self.frames_since_last_shot = self.MIN_FRAMES_BETWEEN_SHOTS

#         self.results = []

#     def update(self, probs, frame_id):
#         print(f"Frame {frame_id}: Probabilities: {probs}")  # Debug

#         if len(probs) == 5:
#             self.probs = probs

#         if probs[0] > 0.8 and self.frames_since_last_shot > self.MIN_FRAMES_BETWEEN_SHOTS:  # Lowered threshold for testing
#             self.nb_cross_courts += 1
#             self.last_shot = "cross_court"
#             self.frames_since_last_shot = 0
#             self.results.append({"FrameID": frame_id, "Shot": self.last_shot})
#             print(f"Shot Detected: {self.last_shot} at Frame {frame_id}")  # Debug
#         elif probs[1] > 0.8 and self.frames_since_last_shot > self.MIN_FRAMES_BETWEEN_SHOTS:
#             self.nb_boasts += 1
#             self.last_shot = "boast"
#             self.frames_since_last_shot = 0
#             self.results.append({"FrameID": frame_id, "Shot": self.last_shot})
#             print(f"Shot Detected: {self.last_shot} at Frame {frame_id}")
#         elif probs[2] > 0.8 and self.frames_since_last_shot > self.MIN_FRAMES_BETWEEN_SHOTS:
#             self.nb_rails += 1
#             self.last_shot = "rail"
#             self.frames_since_last_shot = 0
#             self.results.append({"FrameID": frame_id, "Shot": self.last_shot})
#             print(f"Shot Detected: {self.last_shot} at Frame {frame_id}")
#         elif probs[3] > 0.8 and self.frames_since_last_shot > self.MIN_FRAMES_BETWEEN_SHOTS:
#             self.nb_serves += 1
#             self.last_shot = "serve"
#             self.frames_since_last_shot = 0
#             self.results.append({"FrameID": frame_id, "Shot": self.last_shot})
#             print(f"Shot Detected: {self.last_shot} at Frame {frame_id}")
#         elif probs[4] > 0.8 and self.frames_since_last_shot > self.MIN_FRAMES_BETWEEN_SHOTS:
#             self.nb_no_shots += 1
#             self.last_shot = "no_shot"
#             self.frames_since_last_shot = 0
#             self.results.append({"FrameID": frame_id, "Shot": self.last_shot})
#             print(f"Shot Detected: {self.last_shot} at Frame {frame_id}")

#         self.frames_since_last_shot += 1

#     def display(self, frame):
#         cv2.putText(frame, f"Cross Court = {self.nb_cross_courts}", (20, frame.shape[0] - 140),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if (self.last_shot == "cross_court" and self.frames_since_last_shot < 30) else (0, 0, 255), 2)
#         cv2.putText(frame, f"Boast = {self.nb_boasts}", (20, frame.shape[0] - 110),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if (self.last_shot == "boast" and self.frames_since_last_shot < 30) else (0, 0, 255), 2)
#         cv2.putText(frame, f"Rail = {self.nb_rails}", (20, frame.shape[0] - 80),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if (self.last_shot == "rail" and self.frames_since_last_shot < 30) else (0, 0, 255), 2)
#         cv2.putText(frame, f"Serve = {self.nb_serves}", (20, frame.shape[0] - 50),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if (self.last_shot == "serve" and self.frames_since_last_shot < 30) else (0, 0, 255), 2)
#         cv2.putText(frame, f"No Shot = {self.nb_no_shots}", (20, frame.shape[0] - 20),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if (self.last_shot == "no_shot" and self.frames_since_last_shot < 30) else (0, 0, 255), 2)

# def draw_frame_id(frame, frame_id):
#     cv2.putText(frame, f"Frame {frame_id}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

# if __name__ == "__main__":
#     parser = ArgumentParser(description="Track tennis player and display shot probabilities")
#     parser.add_argument("video")
#     parser.add_argument("model")
#     parser.add_argument("--output", default="output.mp4", help="Output video filename")
#     args = parser.parse_args()

#     shot_counter = ShotCounter()

#     m1 = keras.models.load_model(args.model)

#     cap = cv2.VideoCapture(args.video)
#     assert cap.isOpened()

#     ret, frame = cap.read()
#     assert frame is not None

#     frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = int(cap.get(cv2.CAP_PROP_FPS))
#     out = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

#     human_pose_extractor = HumanPoseExtractor(frame.shape)

#     NB_IMAGES = 30
#     FRAME_ID = 0
#     features_pool = []

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         FRAME_ID += 1

#         human_pose_extractor.extract(frame)
#         human_pose_extractor.discard(["left_eye", "right_eye", "left_ear", "right_ear"])
#         features = human_pose_extractor.keypoints_with_scores.reshape(17, 3)
#         features = features[features[:, 2] > 0][:, 0:2].reshape(1, 13 * 2)
#         features_pool.append(features)

#         if len(features_pool) == NB_IMAGES:
#             features_seq = np.array(features_pool).reshape(1, NB_IMAGES, 26)
#             assert features_seq.shape == (1, 30, 26)
            
#             print(f"Features Sequence: {features_seq.shape}")  # Debug
            
#             probs = (
#                 m1.__call__(features_seq)[0] if human_pose_extractor.roi.valid else np.zeros(5)
#             )
            
#             print(f"ROI Validity: {human_pose_extractor.roi.valid}")  # Debug
#             print(f"Model Probabilities: {probs}")  # Debug
            
#             shot_counter.update(probs, FRAME_ID)
#             features_pool = features_pool[1:]

#         shot_counter.display(frame)
#         draw_frame_id(frame, FRAME_ID)
#         human_pose_extractor.draw_results_frame(frame)

#         out.write(frame)

#     cap.release()
#     out.release()
#     cv2.destroyAllWindows()


##############

import time
from argparse import ArgumentParser
import tensorflow as tf
import numpy as np
import cv2
from tensorflow import keras
from annotation_visualization import HumanPoseExtractor

class ShotCounter:
    MIN_FRAMES_BETWEEN_SHOTS = 60

    def __init__(self):
        self.nb_history = 30
        self.probs = np.zeros(5)

        self.nb_cross_courts = 0
        self.nb_boasts = 0
        self.nb_rails = 0
        self.nb_serves = 0
        self.nb_no_shots = 0

        self.last_shot = "neutral"
        self.frames_since_last_shot = self.MIN_FRAMES_BETWEEN_SHOTS

        self.results = []

    def update(self, probs, frame_id):
        #print(f"Frame {frame_id}: Probabilities: {probs}")  # Debug

        if len(probs) == 5:
            self.probs = probs
        else: 
            self.probs[0:4]=probs
        means = np.mean(self.probs, axis=0)
        if probs[0] > 0.3 and self.frames_since_last_shot > self.MIN_FRAMES_BETWEEN_SHOTS:  # Lowered threshold for testing
            self.nb_cross_courts += 1
            self.last_shot = "cross_court"
            self.frames_since_last_shot = 0
            self.results.append({"FrameID": frame_id, "Shot": self.last_shot})
            print(f"Shot Detected: {self.last_shot} at Frame {frame_id}")  # Debug
        elif probs[1] > 0.3 and self.frames_since_last_shot > self.MIN_FRAMES_BETWEEN_SHOTS:
            self.nb_boasts += 1
            self.last_shot = "boast"
            self.frames_since_last_shot = 0
            self.results.append({"FrameID": frame_id, "Shot": self.last_shot})
            print(f"Shot Detected: {self.last_shot} at Frame {frame_id}")
        elif probs[2] > 0.3 and self.frames_since_last_shot > self.MIN_FRAMES_BETWEEN_SHOTS:
            self.nb_rails += 1
            self.last_shot = "rail"
            self.frames_since_last_shot = 0
            self.results.append({"FrameID": frame_id, "Shot": self.last_shot})
            print(f"Shot Detected: {self.last_shot} at Frame {frame_id}")
        elif probs[3] > 0.3 and self.frames_since_last_shot > self.MIN_FRAMES_BETWEEN_SHOTS:
            self.nb_serves += 1
            self.last_shot = "serve"
            self.frames_since_last_shot = 0
            self.results.append({"FrameID": frame_id, "Shot": self.last_shot})
            print(f"Shot Detected: {self.last_shot} at Frame {frame_id}")
        elif probs[4] > 0.3 and self.frames_since_last_shot > self.MIN_FRAMES_BETWEEN_SHOTS:
            self.nb_no_shots += 1
            self.last_shot = "no_shot"
            self.frames_since_last_shot = 0
            self.results.append({"FrameID": frame_id, "Shot": self.last_shot})
            print(f"Shot Detected: {self.last_shot} at Frame {frame_id}")

        self.frames_since_last_shot += 1

    def display(self, frame):
        cv2.putText(frame, f"Cross Court = {self.nb_cross_courts}", (20, frame.shape[0] - 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if (self.last_shot == "cross_court" and self.frames_since_last_shot < 30) else (0, 0, 255), 2)
        cv2.putText(frame, f"Boast = {self.nb_boasts}", (20, frame.shape[0] - 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if (self.last_shot == "boast" and self.frames_since_last_shot < 30) else (0, 0, 255), 2)
        cv2.putText(frame, f"Rail = {self.nb_rails}", (20, frame.shape[0] - 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if (self.last_shot == "rail" and self.frames_since_last_shot < 30) else (0, 0, 255), 2)
        cv2.putText(frame, f"Serve = {self.nb_serves}", (20, frame.shape[0] - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if (self.last_shot == "serve" and self.frames_since_last_shot < 30) else (0, 0, 255), 2)
        cv2.putText(frame, f"No Shot = {self.nb_no_shots}", (20, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if (self.last_shot == "no_shot" and self.frames_since_last_shot < 30) else (0, 0, 255), 2)


def draw_frame_id(frame, frame_id):
    cv2.putText(frame, f"Frame {frame_id}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

if __name__ == "__main__":
    parser = ArgumentParser(description="Track tennis player and display shot probabilities")
    parser.add_argument("video")
    parser.add_argument("model")
    parser.add_argument("--output", default="output.mp4", help="Output video filename")
    args = parser.parse_args()

    shot_counter = ShotCounter()

    m1 = keras.models.load_model(args.model)

    cap = cv2.VideoCapture(args.video)
    assert cap.isOpened()

    ret, frame = cap.read()
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    human_pose_extractor = HumanPoseExtractor(frame.shape)

    NB_IMAGES = 30
    FRAME_ID = 0
    features_pool = []
    prev_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        FRAME_ID += 1

        human_pose_extractor.extract(frame)
        human_pose_extractor.discard(["left_eye", "right_eye", "left_ear", "right_ear"])

        features = human_pose_extractor.keypoints_with_scores.reshape(17, 3)
        features = features[features[:, 2] > 0][:, 0:2].reshape(1, 13 * 2)
        features_pool.append(features)

        if len(features_pool) == NB_IMAGES:
            features_seq = np.array(features_pool).reshape(1, NB_IMAGES, 26)
            assert features_seq.shape == (1, 30, 26)
            probs = (
                m1.__call__(features_seq)[0] if human_pose_extractor.roi.valid else np.zeros(5)
            )
            shot_counter.update(probs, FRAME_ID)
            features_pool = features_pool[1:]

        shot_counter.display(frame)

        draw_frame_id(frame, FRAME_ID)

        human_pose_extractor.draw_results_frame(frame)

        # Ensure the roi is updated with the keypoints
        human_pose_extractor.roi.update(human_pose_extractor.keypoints_pixels_frame)

        # Write the processed frame to the output video
        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
