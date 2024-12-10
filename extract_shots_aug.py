# ############################################################################
# from argparse import ArgumentParser
# from pathlib import Path
# import numpy as np
# import cv2
# import pandas as pd
# import os
# from extract_human_pose import HumanPoseExtractor

# # Define columns for output CSVs
# columns = [
#     "nose_y", "nose_x", "left_shoulder_y", "left_shoulder_x", "right_shoulder_y", "right_shoulder_x",
#     "left_elbow_y", "left_elbow_x", "right_elbow_y", "right_elbow_x", "left_wrist_y", "left_wrist_x",
#     "right_wrist_y", "right_wrist_x", "left_hip_y", "left_hip_x", "right_hip_y", "right_hip_x",
#     "left_knee_y", "left_knee_x", "right_knee_y", "right_knee_x", "left_ankle_y", "left_ankle_x",
#     "right_ankle_y", "right_ankle_x", 
# ]

# # Utility to ensure directory exists
# def ensure_dir(file_path):
#     directory = os.path.dirname(file_path)
#     if not os.path.exists(directory):
#         os.makedirs(directory)

# # Utility to draw shot name on frame
# def draw_shot(frame, shot):
#     cv2.putText(frame, shot, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(0, 165, 255), thickness=2)
#     print(f"Capturing {shot}")

# # Main script
# if __name__ == "__main__":
#     parser = ArgumentParser(description="Annotate (associate human pose to a squash shot)")
#     parser.add_argument("video", type=str, help="Path to the video file")
#     parser.add_argument("annotation", type=str, help="Path to the annotation CSV file")
#     parser.add_argument("out", type=str, help="Output directory to save shot CSVs")
#     parser.add_argument("--show", action="store_true", help="Show frame")
#     args = parser.parse_args()

#     # Create output directory
#     os.makedirs(args.out, exist_ok=True)

#     # Load annotations
#     shots = pd.read_csv(args.annotation)

#     # Initialize variables
#     CURRENT_ROW = 0
#     NB_IMAGES = 30
#     shots_features = []

#     FRAME_ID = 1
#     IDX_RAIL = 1
#     IDX_BOAST = 1
#     IDX_CROSS_COURT = 1
#     IDX_SERVE = 1
#     no_shot_idx = 1  # Counter for no_shot CSVs

#     cap = cv2.VideoCapture(args.video)
#     assert cap.isOpened(), "Error opening video file."

#     ret, frame = cap.read()
#     human_pose_extractor = HumanPoseExtractor(frame.shape)

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         if CURRENT_ROW >= len(shots):
#             print("Done, no more shots in annotation!")
#             break

#         human_pose_extractor.extract(frame)

#         # Discard insignificant points
#         human_pose_extractor.discard(["left_eye", "right_eye", "left_ear", "right_ear"])
#         features = human_pose_extractor.keypoints_with_scores.reshape(17, 3)

#         if shots.iloc[CURRENT_ROW]["FrameId"] - NB_IMAGES // 2 == FRAME_ID:
#             shots_features = []

#         # Handle regular shot classes
#         if (
#             shots.iloc[CURRENT_ROW]["FrameId"] - NB_IMAGES // 2
#             <= FRAME_ID
#             <= shots.iloc[CURRENT_ROW]["FrameId"] + NB_IMAGES // 2
#         ):
#             if np.mean(features[:, 2]) < 0.1:
#                 CURRENT_ROW += 1
#                 shots_features = []
#                 print("Cancel this shot")
#                 FRAME_ID += 1
#                 continue

#             features = features[features[:, 2] > 0][:, 0:2].reshape(1, 13 * 2)

#             shot_class = shots.iloc[CURRENT_ROW]["Shot"]
#             shots_features.append(features)
#             draw_shot(frame, shot_class)

#             if FRAME_ID - NB_IMAGES // 2 + 1 == shots.iloc[CURRENT_ROW]["FrameId"]:
#                 if len(shots_features) == NB_IMAGES:
#                     # Save original frames first
#                     shots_df_original = pd.DataFrame(np.concatenate(shots_features, axis=0),columns=columns,)
#                     shots_df_original["shot"] = np.full(NB_IMAGES, shot_class)

#                     # Save the original shot
#                     if shot_class == "rail":
#                         outpath_original = Path(args.out).joinpath(f"rail_{IDX_RAIL:03d}.csv")
#                         IDX_RAIL += 1
#                     elif shot_class == "cross_court":
#                         outpath_original = Path(args.out).joinpath(f"cross_court_{IDX_CROSS_COURT:03d}.csv")
#                         IDX_CROSS_COURT += 1
#                     elif shot_class == "boast":
#                         outpath_original = Path(args.out).joinpath(f"boast_{IDX_BOAST:03d}.csv")
#                         IDX_BOAST += 1
#                     elif shot_class == "serve":
#                         outpath_original = Path(args.out).joinpath(f"serve_{IDX_SERVE:03d}.csv")
#                         IDX_SERVE += 1

#                     shots_df_original.to_csv(outpath_original, index=False)
#                     print(f"saving original 30 frames for {shot_class} to {outpath_original}")

#                     # Apply jitter: Replace first 3 frames with jittered frames in one CSV
#                     jittered_features_first_3 = []
#                     for i in range(3):
#                         jittered_frame = shots_features[i].copy()
#                         jittered_frame += np.random.uniform(-0.01, 0.01, size=jittered_frame.shape)
#                         jittered_features_first_3.append(jittered_frame)
#                         shots_features[i] = jittered_frame  # Replace with jittered frame

#                     # Save all 30 frames with jittered first 3
#                     shots_df_first_3 = pd.DataFrame(
#                         np.concatenate(shots_features, axis=0),
#                         columns=columns,
#                     )
#                     shots_df_first_3["shot"] = np.full(NB_IMAGES, shot_class)

#                     # Save CSV with first 3 jittered frames
#                     if shot_class == "rail":
#                         outpath_first_3 = Path(args.out).joinpath(f"rail_{IDX_RAIL:03d}.csv")
#                         IDX_RAIL += 1
#                     elif shot_class == "cross_court":
#                         outpath_first_3 = Path(args.out).joinpath(f"cross_court_{IDX_CROSS_COURT:03d}.csv")
#                         IDX_CROSS_COURT += 1
#                     elif shot_class == "boast":
#                         outpath_first_3 = Path(args.out).joinpath(f"boast_{IDX_BOAST:03d}.csv")
#                         IDX_BOAST += 1
#                     elif shot_class == "serve":
#                         outpath_first_3 = Path(args.out).joinpath(f"serve_{IDX_SERVE:03d}.csv")
#                         IDX_SERVE += 1

#                     shots_df_first_3.to_csv(outpath_first_3, index=False)
#                     print(f"saving 30 frames (with first 3 jittered) for {shot_class} to {outpath_first_3}")

#                     # Replace last 3 frames with jittered frames in the next CSV
#                     jittered_features_last_3 = []
#                     for i in range(27, 30):
#                         jittered_frame = shots_features[i].copy()
#                         jittered_frame += np.random.uniform(-0.01, 0.01, size=jittered_frame.shape)
#                         jittered_features_last_3.append(jittered_frame)
#                         shots_features[i] = jittered_frame  # Replace with jittered frame

#                     # Save all 30 frames with jittered last 3
#                     shots_df_last_3 = pd.DataFrame(
#                         np.concatenate(shots_features, axis=0),
#                         columns=columns,
#                     )
#                     shots_df_last_3["shot"] = np.full(NB_IMAGES, shot_class)

#                     # Save CSV with last 3 jittered frames
#                     if shot_class == "rail":
#                         outpath_last_3 = Path(args.out).joinpath(f"rail_{IDX_RAIL:03d}.csv")
#                         IDX_RAIL += 1
#                     elif shot_class == "cross_court":
#                         outpath_last_3 = Path(args.out).joinpath(f"cross_court_{IDX_CROSS_COURT:03d}.csv")
#                         IDX_CROSS_COURT += 1
#                     elif shot_class == "boast":
#                         outpath_last_3 = Path(args.out).joinpath(f"boast_{IDX_BOAST:03d}.csv")
#                         IDX_BOAST += 1
#                     elif shot_class == "serve":
#                         outpath_last_3 = Path(args.out).joinpath(f"serve_{IDX_SERVE:03d}.csv")
#                         IDX_SERVE += 1

#                     shots_df_last_3.to_csv(outpath_last_3, index=False)
#                     print(f"saving 30 frames (with last 3 jittered) for {shot_class} to {outpath_last_3}")

#                     shots_features = []  # Reset features for the next shot
#                     CURRENT_ROW += 1
#         elif (
#                     shots.iloc[CURRENT_ROW]["FrameId"] - shots.iloc[CURRENT_ROW - 1]["FrameId"]
#                     > NB_IMAGES
#                 ):
#                     frame_id_between_shots = (
#                         shots.iloc[CURRENT_ROW - 1]["FrameId"]
#                         + shots.iloc[CURRENT_ROW]["FrameId"]
#                     ) // 2
#                     if (
#                         frame_id_between_shots - NB_IMAGES // 2
#                         < FRAME_ID
#                         <= frame_id_between_shots + NB_IMAGES // 2
#                     ):
#                         features = features[features[:, 2] > 0][:, 0:2].reshape(1, 13 * 2)
#                         shots_features.append(features)
#                         draw_shot(frame, "no_shot")

#                         if FRAME_ID == frame_id_between_shots + NB_IMAGES // 2:
#                             if len(shots_features) == NB_IMAGES:
#                                 shots_df = pd.DataFrame(
#                                     np.concatenate(shots_features, axis=0),
#                                     columns=columns,
#                                 )
#                                 shots_df["shot"] = np.full(NB_IMAGES, "no_shot")
#                                 outpath = Path(args.out).joinpath(f"no_shot_{no_shot_idx:03d}.csv")
#                                 print(f"saving no_shot to {outpath}")
#                                 no_shot_idx += 1
#                                 shots_df.to_csv(outpath, index=False)
#                             else:
#                                 print(f"Skipping no_shot, insufficient frames: {len(shots_features)}")
#                             shots_features = []

#         if args.show:
#             cv2.imshow("frame", frame)
#             if cv2.waitKey(1) == 27:  # Press 'ESC' to quit
#                 break

#         FRAME_ID += 1

#     cap.release()
#     cv2.destroyAllWindows()

############################################################################
from argparse import ArgumentParser
from pathlib import Path
import numpy as np
import cv2
import pandas as pd
import os
from annotation_visualization import HumanPoseExtractor

# Define columns for output CSVs
columns = [
    "nose_y", "nose_x", "left_shoulder_y", "left_shoulder_x", "right_shoulder_y", "right_shoulder_x",
    "left_elbow_y", "left_elbow_x", "right_elbow_y", "right_elbow_x", "left_wrist_y", "left_wrist_x",
    "right_wrist_y", "right_wrist_x", "left_hip_y", "left_hip_x", "right_hip_y", "right_hip_x",
    "left_knee_y", "left_knee_x", "right_knee_y", "right_knee_x", "left_ankle_y", "left_ankle_x",
    "right_ankle_y", "right_ankle_x", 
]

# Utility to ensure directory exists
def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

# Utility to draw shot name on frame
def draw_shot(frame, shot):
    cv2.putText(frame, shot, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(0, 165, 255), thickness=2)
    print(f"Capturing {shot}")

# Add timestamps to saved CSVs
def add_timestamps_to_csvs(output_dir, fps):
    output_dir = Path(output_dir)
    csv_files = list(output_dir.glob("*.csv"))
    print(f"Found {len(csv_files)} CSV files to process for timestamps.")
    
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        num_frames = len(df)
        timestamps = [(i / fps) for i in range(num_frames)]
        df["timestamp"] = timestamps
        df.to_csv(csv_file, index=False)
        print(f"Timestamps added to {csv_file}")

# Main script
if __name__ == "__main__":
    parser = ArgumentParser(description="Annotate (associate human pose to a squash shot)")
    parser.add_argument("video", type=str, help="Path to the video file")
    parser.add_argument("annotation", type=str, help="Path to the annotation CSV file")
    parser.add_argument("out", type=str, help="Output directory to save shot CSVs")
    parser.add_argument("--show", action="store_true", help="Show frame")
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.out, exist_ok=True)

    # Load annotations
    shots = pd.read_csv(args.annotation)

    # Initialize variables
    CURRENT_ROW = 0
    NB_IMAGES = 30
    shots_features = []

    FRAME_ID = 1
    IDX_RAIL = 1
    IDX_BOAST = 1
    IDX_CROSS_COURT = 1
    IDX_SERVE = 1
    no_shot_idx = 1  # Counter for no_shot CSVs

    cap = cv2.VideoCapture(args.video)
    assert cap.isOpened(), "Error opening video file."


    # Retrieve video fps
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video_FPS: {fps}")
    
    ret, frame = cap.read()
    human_pose_extractor = HumanPoseExtractor(frame.shape)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if CURRENT_ROW >= len(shots):
            print("Done, no more shots in annotation!")
            break

        human_pose_extractor.extract(frame)

        # Discard insignificant points
        human_pose_extractor.discard(["left_eye", "right_eye", "left_ear", "right_ear"])
        features = human_pose_extractor.keypoints_with_scores.reshape(17, 3)

        if shots.iloc[CURRENT_ROW]["FrameId"] - NB_IMAGES // 2 == FRAME_ID:
            shots_features = []

        # Handle regular shot classes
        if (
            shots.iloc[CURRENT_ROW]["FrameId"] - NB_IMAGES // 2
            <= FRAME_ID
            <= shots.iloc[CURRENT_ROW]["FrameId"] + NB_IMAGES // 2
        ):
            if np.mean(features[:, 2]) < 0.1:
                CURRENT_ROW += 1
                shots_features = []
                print("Cancel this shot")
                FRAME_ID += 1
                continue

            features = features[features[:, 2] > 0][:, 0:2].reshape(1, 13 * 2)

            shot_class = shots.iloc[CURRENT_ROW]["Shot"]
            shots_features.append(features)
            draw_shot(frame, shot_class)

            if FRAME_ID - NB_IMAGES // 2 + 1 == shots.iloc[CURRENT_ROW]["FrameId"]:
                if len(shots_features) == NB_IMAGES:
                    # Save original frames first
                    shots_df_original = pd.DataFrame(np.concatenate(shots_features, axis=0),columns=columns,)
                    shots_df_original["shot"] = np.full(NB_IMAGES, shot_class)

                    # Save the original shot
                    if shot_class == "rail":
                        outpath_original = Path(args.out).joinpath(f"rail_{IDX_RAIL:03d}.csv")
                        IDX_RAIL += 1
                    elif shot_class == "cross_court":
                        outpath_original = Path(args.out).joinpath(f"cross_court_{IDX_CROSS_COURT:03d}.csv")
                        IDX_CROSS_COURT += 1
                    elif shot_class == "boast":
                        outpath_original = Path(args.out).joinpath(f"boast_{IDX_BOAST:03d}.csv")
                        IDX_BOAST += 1
                    elif shot_class == "serve":
                        outpath_original = Path(args.out).joinpath(f"serve_{IDX_SERVE:03d}.csv")
                        IDX_SERVE += 1

                    shots_df_original.to_csv(outpath_original, index=False)
                    print(f"saving {shot_class} to {outpath_original}")

                    # Apply jitter: Replace first 3 frames with jittered frames in one CSV
                    jittered_features_first_3 = []
                    for i in range(3):
                        jittered_frame = shots_features[i].copy()
                        jittered_frame += np.random.uniform(-0.01, 0.01, size=jittered_frame.shape)
                        jittered_features_first_3.append(jittered_frame)
                        shots_features[i] = jittered_frame  # Replace with jittered frame

                    # Save all 30 frames with jittered first 3
                    shots_df_first_3 = pd.DataFrame(
                        np.concatenate(shots_features, axis=0),
                        columns=columns,
                    )
                    shots_df_first_3["shot"] = np.full(NB_IMAGES, shot_class)

                    # Save CSV with first 3 jittered frames
                    if shot_class == "rail":
                        outpath_first_3 = Path(args.out).joinpath(f"rail_{IDX_RAIL:03d}.csv")
                        IDX_RAIL += 1
                    elif shot_class == "cross_court":
                        outpath_first_3 = Path(args.out).joinpath(f"cross_court_{IDX_CROSS_COURT:03d}.csv")
                        IDX_CROSS_COURT += 1
                    elif shot_class == "boast":
                        outpath_first_3 = Path(args.out).joinpath(f"boast_{IDX_BOAST:03d}.csv")
                        IDX_BOAST += 1
                    elif shot_class == "serve":
                        outpath_first_3 = Path(args.out).joinpath(f"serve_{IDX_SERVE:03d}.csv")
                        IDX_SERVE += 1

                    shots_df_first_3.to_csv(outpath_first_3, index=False)
                    print(f"saving 30 frames (with first 3 jittered) for {shot_class} to {outpath_first_3}")

                    # Replace last 3 frames with jittered frames in the next CSV
                    jittered_features_last_3 = []
                    for i in range(27, 30):
                        jittered_frame = shots_features[i].copy()
                        jittered_frame += np.random.uniform(-0.01, 0.01, size=jittered_frame.shape)
                        jittered_features_last_3.append(jittered_frame)
                        shots_features[i] = jittered_frame  # Replace with jittered frame

                    # Save all 30 frames with jittered last 3
                    shots_df_last_3 = pd.DataFrame(
                        np.concatenate(shots_features, axis=0),
                        columns=columns,
                    )
                    shots_df_last_3["shot"] = np.full(NB_IMAGES, shot_class)

                    # Save CSV with last 3 jittered frames
                    if shot_class == "rail":
                        outpath_last_3 = Path(args.out).joinpath(f"rail_{IDX_RAIL:03d}.csv")
                        IDX_RAIL += 1
                    elif shot_class == "cross_court":
                        outpath_last_3 = Path(args.out).joinpath(f"cross_court_{IDX_CROSS_COURT:03d}.csv")
                        IDX_CROSS_COURT += 1
                    elif shot_class == "boast":
                        outpath_last_3 = Path(args.out).joinpath(f"boast_{IDX_BOAST:03d}.csv")
                        IDX_BOAST += 1
                    elif shot_class == "serve":
                        outpath_last_3 = Path(args.out).joinpath(f"serve_{IDX_SERVE:03d}.csv")
                        IDX_SERVE += 1

                    shots_df_last_3.to_csv(outpath_last_3, index=False)
                    print(f"saving 30 frames (with last 3 jittered) for {shot_class} to {outpath_last_3}")

                    shots_features = []  # Reset features for the next shot
                    CURRENT_ROW += 1
        elif (
                    shots.iloc[CURRENT_ROW]["FrameId"] - shots.iloc[CURRENT_ROW - 1]["FrameId"]
                    > NB_IMAGES
                ):
                    frame_id_between_shots = (
                        shots.iloc[CURRENT_ROW - 1]["FrameId"]
                        + shots.iloc[CURRENT_ROW]["FrameId"]
                    ) // 2
                    if (
                        frame_id_between_shots - NB_IMAGES // 2
                        < FRAME_ID
                        <= frame_id_between_shots + NB_IMAGES // 2
                    ):
                        features = features[features[:, 2] > 0][:, 0:2].reshape(1, 13 * 2)
                        shots_features.append(features)
                        draw_shot(frame, "no_shot")

                        if FRAME_ID == frame_id_between_shots + NB_IMAGES // 2:
                            if len(shots_features) == NB_IMAGES:
                                shots_df = pd.DataFrame(
                                    np.concatenate(shots_features, axis=0),
                                    columns=columns,
                                )
                                shots_df["shot"] = np.full(NB_IMAGES, "no_shot")
                                outpath = Path(args.out).joinpath(f"no_shot_{no_shot_idx:03d}.csv")
                                print(f"saving no_shot to {outpath}")
                                no_shot_idx += 1
                                shots_df.to_csv(outpath, index=False)
                            else:
                                print(f"Skipping no_shot, insufficient frames: {len(shots_features)}")
                            shots_features = []

        if args.show:
            cv2.imshow("frame", frame)
            if cv2.waitKey(1) == 27:  # Press 'ESC' to quit
                break

        FRAME_ID += 1

    cap.release()
    cv2.destroyAllWindows()

    # Add timestamps to saved CSVs
    add_timestamps_to_csvs(args.out, fps)
