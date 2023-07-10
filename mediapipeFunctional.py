# File Created to test MediaPipe Pose Estimator for Functional Assessment

import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt

def file_import(path):
	'''
	Reads in video file in cv2 capture object
	'''
	cap = cv2.VideoCapture(path)

	return cap


def generate_frames_file():
	'''
	Creates Pose Estimates for a Single Video File with output handout
	'''
	filename = "IMG_6746.MOV"
	cap = file_import(filename)

	# Initialize MediaPipe Drawing
	mp_drawing = mp.solutions.drawing_utils

	# Initialize MediaPipe Pose model
	mp_pose = mp.solutions.pose

	start_frame = 0
	end_frame = 0
	threshold_angle_hip = 160
	hip_angle_data = []

	with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
		while cap.isOpened():
			try:

				ret, frame = cap.read()

				# Recolor image to RGB
				image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
				image.flags.writeable = False

				# Make detection
				results = pose.process(image)

				# Recolor back to BGR
				image.flags.writeable = True
				image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

				try:
					landmarks = results.pose_landmarks.landmark

					# Get coordinates Lower Body
					knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
					       landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]

					hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
						   landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]

					# Get coordinates Upper Body
					shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
					            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]

					# Calculate angles
					hip_angle = calculate_angle(knee, hip, shoulder)


					# Update start and end frames
					if hip_angle < threshold_angle_hip and start_frame == 0:
						start_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
					if hip_angle > threshold_angle_hip and start_frame != 0 and end_frame == 0:
						end_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)


					# Crop and store hip angle data within the squat range
					if start_frame != 0 and end_frame == 0:
						hip_angle_data.append(hip_angle)


					# Visualize angle of hip_angle at the hip
					cv2.putText(image, str(hip_angle),
					            tuple(np.multiply(hip, [640, 480]).astype(int)),
					            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2,
					            cv2.LINE_AA)


				except:
					pass

				# Render detections
				mp_drawing.draw_landmarks(image, results.pose_landmarks,
				                          mp_pose.POSE_CONNECTIONS,
				                          mp_drawing.DrawingSpec(
					                          color=(245, 117, 66),
					                          thickness=2,
					                          circle_radius=2),
				                          mp_drawing.DrawingSpec(
					                          color=(245, 66, 230),
					                          thickness=2,
					                          circle_radius=2)
				                          )

				# Display the image
				cv2.imshow('Squat Video', image)
				if cv2.waitKey(1) & 0xFF == ord('q'):
					break

			except:
				break

		cap.release()
		cv2.destroyAllWindows()

		print(hip_angle_data)
		make_plot(hip_angle_data)


def make_plot(angle_data):
	plt.figure()
	plt.plot(angle_data)
	plt.xlabel('Frame')
	plt.ylabel('Hip Angle (degrees)')
	plt.title('Hip Angle during Squat')
	plt.show()

def generate_frames():
	# Initialize MediaPipe Drawing
	mp_drawing = mp.solutions.drawing_utils

	# Initialize MediaPipe Pose model
	mp_pose = mp.solutions.pose

	cap = cv2.VideoCapture(0)
	## Setup mediapipe instance
	with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
		while cap.isOpened():
			ret, frame = cap.read()

			# Recolor image to RGB
			image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			image.flags.writeable = False

			# Make detection
			results = pose.process(image)

			# Recolor back to BGR
			image.flags.writeable = True
			image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

			# Extract landmarks
			try:
				landmarks = results.pose_landmarks.landmark


				# Get coordinates Lower Body
				knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
				       landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]

				hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
					   landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]

				ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
				       landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

				# Get coordinates Upper Body
				shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
				            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]


				# Calculate angles
				angle_trunk = calculate_angle_horz(shoulder, hip)

				angle_shank = calculate_angle_horz(knee, ankle)


				# Visualize angle of trunk and shank on video wrt ground
				cv2.putText(image, str(angle_trunk),
				            tuple(np.multiply(hip, [640, 480]).astype(int)),
				            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2,
				            cv2.LINE_AA)

				cv2.putText(image, str(angle_shank),
				            tuple(np.multiply(knee, [640, 480]).astype(int)),
				            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2,
				            cv2.LINE_AA)

			except:
				pass

			# Render detections
			mp_drawing.draw_landmarks(image, results.pose_landmarks,
			                          mp_pose.POSE_CONNECTIONS,
			                          mp_drawing.DrawingSpec(color=(245, 117, 66),
			                                                 thickness=2,
			                                                 circle_radius=2),
			                          mp_drawing.DrawingSpec(color=(245, 66, 230),
			                                                 thickness=2,
			                                                 circle_radius=2)
			                          )

			cv2.imshow('Mediapipe Feed', image)

			if cv2.waitKey(10) & 0xFF == ord('q'):
				break

		cap.release()
		cv2.destroyAllWindows()

def calculate_angle(point1, point2, point3):
    """Calculate angle between two lines given three points"""
    if point1 == (0, 0) or point2 == (0, 0) or point3 == (0, 0):
        return 0

    # Calculate vectors
    vector1 = (point1[0] - point2[0], point1[1] - point2[1])
    vector2 = (point3[0] - point2[0], point3[1] - point2[1])

    # Calculate dot product
    dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]

    # Calculate magnitudes
    magnitude1 = np.sqrt(vector1[0] ** 2 + vector1[1] ** 2)
    magnitude2 = np.sqrt(vector2[0] ** 2 + vector2[1] ** 2)

    try:
        # Calculate cosine of the angle
        cosine_angle = dot_product / (magnitude1 * magnitude2)

        # Calculate angle in radians
        radian_angle = np.arccos(cosine_angle)

        # Convert angle to degrees
        degree_angle = np.degrees(radian_angle)

        if degree_angle > 180.0:
            degree_angle = 360 - degree_angle

        return degree_angle

    except ZeroDivisionError:
        return 90.0

def calculate_angle_horz(point1, point2):
    """Calculate angle between lines and ground"""
    if point1 == (0, 0) or point2 == (0, 0):
        return 0

    # Calculate vector
    vector1 = (point1[0] - point2[0], point1[1] - point2[1])
    vector2 = (1, 0)

    # Calculate dot product
    dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]

    # Calculate magnitudes
    magnitude1 = np.sqrt(vector1[0] ** 2 + vector1[1] ** 2)
    magnitude2 = np.sqrt(vector2[0] ** 2 + vector2[1] ** 2)

    try:
        # Calculate cosine of the angle
        cosine_angle = dot_product / (magnitude1 * magnitude2)

        # Calculate angle in radians
        radian_angle = np.arccos(cosine_angle)

        # Convert angle to degrees
        degree_angle = np.degrees(radian_angle)

        if degree_angle > 180.0:
            degree_angle = 360 - degree_angle

        return degree_angle

    except ZeroDivisionError:
        return 90.0

if __name__ == '__main__':
    generate_frames_file()