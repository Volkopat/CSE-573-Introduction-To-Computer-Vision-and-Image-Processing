'''
Please do NOT make any changes to this file.
'''

from UBFaceDetector import cluster_faces
import cv2
import numpy as np
import argparse
import json
import os
import sys

def parse_args():
    parser = argparse.ArgumentParser(description="cse 473/573 project 3.")
    parser.add_argument(
        "--input_path", type=str, default="faceCluster_5",
        help="path to faceCluster K folder")
    parser.add_argument(
        "--num_cluster", type=str, default="5",
        help="number of clusters")
    parser.add_argument(
        "--output", type=str, default="./clusters.json",
        help="path to the characters folder")

    args = parser.parse_args()
    return args

def save_results(result_dict, filename):
    results = []
    results = result_dict
    with open(filename, "w") as file:
        json.dump(results, file)

def main():
    if cv2.__version__ != '4.5.4':
        print("Please use OpenCV 4.5.4")
        sys.exit(1)
    args = parse_args()
    path, filename = os.path.split(args.output)
    os.makedirs(path, exist_ok=True)
    result_list = cluster_faces(args.input_path, K=args.num_cluster)
    save_results(result_list, args.output)

if __name__ == "__main__":
    main()

    