# This python module was developed to store the alternative version of
# generate-frames that a live video feed from a camera accessed by cv2,
# this is then used with the flask app

# Crated by Ben Randoing for L-NewCo during 07/2023



##################################################################
################# Live Feed ######################################
##################################################################
def generate_frames():
	# Initialize MediaPipe Drawing
	mp_drawing = mp.solutions.drawing_utils

	# Initialize MediaPipe Pose model
	mp_pose = mp.solutions.pose

	cap = cv2.VideoCapture(0)
	## Setup mediapipe instance
	with mp_pose.Pose(min_detection_confidence=0.5,
	                  min_tracking_confidence=0.5) as pose:
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
				shoulder = [
					landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
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
			                          mp_drawing.DrawingSpec(
				                          color=(245, 117, 66),
				                          thickness=2,
				                          circle_radius=2),
			                          mp_drawing.DrawingSpec(
				                          color=(245, 66, 230),
				                          thickness=2,
				                          circle_radius=2)
			                          )

			cv2.imshow('Mediapipe Feed', image)

			if cv2.waitKey(10) & 0xFF == ord('q'):
				break

		cap.release()
		cv2.destroyAllWindows()
