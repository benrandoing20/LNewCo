# File Created to test MediaPipe Pose Estimator for Functional Assessment

import cv2
import mediapipe as mp
import numpy as np

def run():
	# Initialize MediaPipe Drawing
	mp_drawing = mp.solutions.drawing_utils

	# Initialize MediaPipe Pose model
	mp_pose = mp.solutions.pose

	# Curl counter variables
	counter = 0
	stage = None

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
				elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
				         landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
				wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
				         landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

				# Calculate angle
				angle = calculate_angle(shoulder, elbow, wrist)

				angle_trunk = calculate_angle_horz(shoulder, hip)

				angle_shank = calculate_angle_horz(knee, ankle)

				# Visualize angle
				cv2.putText(image, str(angle),
				            tuple(np.multiply(elbow, [640, 480]).astype(int)),
				            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2,
				            cv2.LINE_AA
				            )

				# Curl counter logic
				if angle > 160:
					stage = "down"
				if angle < 30 and stage == 'down':
					stage = "up"
					counter += 1
					print(counter)

			except:
				pass

			# # Displays landmark names for manipulation
			# for lndmrk in mp_pose.PoseLandmark:
			# 	print(lndmrk)

			# Render curl counter
			# Setup status box
			cv2.rectangle(image, (0, 0), (225, 73), (245, 117, 16), -1)

			# Rep data
			cv2.putText(image, 'REPS', (15, 12),
			            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,
			            cv2.LINE_AA)
			cv2.putText(image, str(counter),
			            (10, 60),
			            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2,
			            cv2.LINE_AA)

			# Stage data
			cv2.putText(image, 'STAGE', (65, 12),
			            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,
			            cv2.LINE_AA)
			cv2.putText(image, stage,
			            (60, 60),
			            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2,
			            cv2.LINE_AA)

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
    run()