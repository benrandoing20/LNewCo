# File Created to test OpenPose for Functional Assessment Automation

import cv2
import time
import argparse
from sys import platform
import numpy as np

# Import OpenPose
try:
    # Change these paths according to your OpenPose installation
    sys.path.append('your_openpose_python_path')
    from openpose import pyopenpose as op
except ImportError as e:
    print(f'Error: OpenPose library could not be found. Did you set the paths correctly?\n{e}')
    sys.exit(-1)

def main():
    # Set the OpenPose parameters
    params = {}
    params["model_folder"] = "your_openpose_models_path"
    params["net_resolution"] = "320x176"  # Change the resolution according to needs

    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    # Start capturing the live video stream
    cap = cv2.VideoCapture(0)  # Change the index if you have multiple cameras

    while True:
        # Read the frame from the video stream
        ret, frame = cap.read()

        if not ret:
            break

        # Process the frame with OpenPose
        datum = op.Datum()
        datum.cvInputData = frame
        opWrapper.emplaceAndPop([datum])

        # Retrieve the pose keypoints
        keypoints = datum.poseKeypoints

        # Render the keypoints on the frame
        if len(keypoints.shape) > 1:
            for person in keypoints:
                for i, body_part in enumerate(person):
                    if body_part[2] > 0.1:  # Only plot keypoints with confidence above a certain threshold
                        cv2.circle(frame, (int(body_part[0]), int(body_part[1])), 4, (0, 255, 0), -1)

        # Display the frame
        cv2.imshow("OpenPose Live Feed", frame)
        if cv2.waitKey(1) == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
