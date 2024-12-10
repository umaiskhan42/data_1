import sqlite3
import pandas as pd
import numpy as np
import os
import cv2
from argparse import ArgumentParser
from pathlib import Path
import pickle

# Utility to ensure directory exists
def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

# Save data to SQLite database
def save_to_database(conn, video_id, offset, shot_type, keypoints):
    """
    Save a shot's data into the SQLite database.
    :param conn: SQLite connection object.
    :param video_id: Name of the video file.
    :param offset: Time offset in seconds for the frame.
    :param shot_type: Type of the shot (rail, boast, etc.).
    :param keypoints: Keypoints as binary data.
    """
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO annotations (Video_id, timestamp, shot_type, keypoints)
        VALUES (?, ?, ?, ?);
    """, (video_id, offset, shot_type, sqlite3.Binary(keypoints)))
    conn.commit()

# Main script
if __name__ == "__main__":
    parser = ArgumentParser(description="Annotate (associate human pose to a squash shot)")
    parser.add_argument("video", type=str, help="Path to the video file")
    parser.add_argument("annotation", type=str, help="Path to the annotation CSV file")
    parser.add_argument("out", type=str, help="Output directory to save shot CSVs")
    parser.add_argument("--show", action="store_true", help="Show frame")
    args = parser.parse_args()

    # Database file path
    db_file = os.path.join(args.out, "annotations.db")

    # Create output directory
    os.makedirs(args.out, exist_ok=True)

    # Connect to SQLite database (create if it doesn't exist)
    conn = sqlite3.connect(db_file)

    # Create table if it doesn't exist
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS annotations (
            Video_id TEXT,
            timestamp REAL,
            shot_type TEXT,
            keypoints BLOB
        );
    """)
    conn.commit()

    # Load annotations
    shots = pd.read_csv(args.annotation)

    # Initialize variables
    CURRENT_ROW = 0
    NB_IMAGES = 30

    cap = cv2.VideoCapture(args.video)
    assert cap.isOpened(), "Error opening video file."

    video_id = Path(args.video).name
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_id = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if CURRENT_ROW >= len(shots):
            print("Done, no more shots in annotation!")
            break

        # Frame ID and offset
        frame_id += 1
        offset = frame_id / fps

        # Example keypoints as numpy array (simulate pose extraction here)
        keypoints = np.random.rand(17, 2).astype(np.float32)  # Replace with actual keypoints extraction

        # Check if the frame is within the shot window
        if (
            shots.iloc[CURRENT_ROW]["FrameId"] - NB_IMAGES // 2
            <= frame_id
            <= shots.iloc[CURRENT_ROW]["FrameId"] + NB_IMAGES // 2
        ):
            shot_type = shots.iloc[CURRENT_ROW]["Shot"]
            keypoints_binary = pickle.dumps(keypoints)  # Serialize keypoints as binary
            
            # Save to database
            save_to_database(conn, video_id, offset, shot_type, keypoints_binary)
            print(f"Saved {shot_type} at {offset:.2f} seconds.")

            if frame_id == shots.iloc[CURRENT_ROW]["FrameId"] + NB_IMAGES // 2:
                CURRENT_ROW += 1

        if args.show:
            cv2.imshow("frame", frame)
            if cv2.waitKey(1) == 27:  # Press 'ESC' to quit
                break

    cap.release()
    conn.close()
    cv2.destroyAllWindows()
