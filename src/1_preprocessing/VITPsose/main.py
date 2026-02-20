import os
import sys
import pandas as pd
import numpy as np
import torch
import cv2
from PIL import Image


import config
import key_points
from video import VideoProcessor
from detectors import HumanDetector, PoseDetector


def construct_path(movie_files):
    movie_paths = []
    output_movie_paths = []
    output_raw_paths = []

    for file in movie_files:
        movie_paths.append(os.path.join(config.MOVIE_DIR, file))
        output_movie_paths.append(os.path.join(config.OUTPUT_MOVIE_DIR, f"detected_{file}"))
        output_raw_paths.append(os.path.join(config.OUTPUT_RAW_DIR, f"detected_{file}").replace(".mp4", ".csv"))
    
    return movie_paths, output_movie_paths, output_raw_paths

def print_progress(current, total, bar_length=40):
    progress = current / total
    filled = int(bar_length * progress)
    bar = "█" * filled + "-" * (bar_length - filled)
    sys.stdout.write(f"\r[{bar}] {current}/{total} ({progress*100:.1f}%)")
    sys.stdout.flush()

def make_df(timestamps, keypoints_array):
    record_df = pd.DataFrame(keypoints_array, columns=key_points.COLUMN_LIST)
    record_df.insert(0, "TIME", timestamps)
    return record_df

def main():
    print("Start Tracking Process Using ViTPose")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device}")
    hd = HumanDetector(device)
    posedetector = PoseDetector(device)

    files = os.listdir(config.MOVIE_DIR)
    movie_paths, output_movie_paths, output_raw_paths = construct_path(files)

    for i, movie_path in enumerate(movie_paths):
        if "test" not in movie_path:
            continue
        print(f"\nMovie {i} : {files[i]}")
        vp = VideoProcessor(movie_path, output_movie_paths[i])

        total_frame = vp.get_total_frame()
        current_frame = 0

        keypoint_list = []
        timestamp_list = []

        while True:
            print_progress(current_frame, total_frame)
            ret, frame = vp.read()

            if not ret:
                break
            timestamp_list.append(vp.get_timestamp())


            rgb_array = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_array)

            hd.process(pil_img)
            person_boxes_xyxy = hd.get_person_boxes()
            if len(person_boxes_xyxy) > 0:
                person_boxes = hd.convert_format(person_boxes_xyxy)

                posedetector.process(pil_img, person_boxes)
                black_img = Image.new(pil_img.mode, pil_img.size, (0, 0, 0, 255))

                annotated_frame = posedetector.visualize_keypoints(black_img, person_boxes_xyxy)
                keypoint_list.append(posedetector.get_keypoints_array())

                cv_annotated_array = cv2.cvtColor(np.array(annotated_frame), cv2.COLOR_RGB2BGR)

                vp.write(cv_annotated_array)
            else:
                keypoint_list.append(np.full(len(key_points.COLUMN_LIST), np.nan))
                cv_annotated_array = cv2.cvtColor(np.array(black_img), cv2.COLOR_RGB2BGR)
                vp.write(cv_annotated_array)

            current_frame += 1
        vp.release()

        record_df = make_df(timestamp_list, keypoint_list)
        record_df.to_csv(output_raw_paths[i], index=False)

if __name__ == "__main__":
    main()