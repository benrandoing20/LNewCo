# File Created to test MediaPipe Pose Estimator for Functional Assessment
# Created by Ben Randoing for L-NewCo during 07/2023

import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from fpdf import FPDF
from PIL import Image
from angle_calc import *
import traceback


def file_import(path):
	'''
	Reads in video file in cv2 capture object
	'''
	cap = cv2.VideoCapture(path)

	return cap


def start_stop(landmarks, mp_pose, cap, start_frame, end_frame, threshold):
	threshold_angle_hip = threshold

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

	return start_frame, end_frame

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
	hip_angle_data = {"L": [], "R": []}
	knee_angle_data = {"L": [], "R": []}
	ankle_angle_data = {"L": [], "R": []}
	deviation_angle_data = []
	threshold = 160
	hip_angle_min = threshold

	global bottom_frame, angle_femur_left, angle_femur_right
	angle_femur_left = None
	angle_femur_right = None
	bottom_frame = None


	with mp_pose.Pose(min_detection_confidence=0.5,
	                  min_tracking_confidence=0.5) as pose:
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

					# Calculate angles
					hip_angle = get_hip_angle(landmarks, mp_pose)
					knee_angle = get_knee_angle(landmarks, mp_pose)
					ankle_angle = get_ankle_angle(landmarks, mp_pose)
					deviation_angle = get_hipshin_angle(landmarks, mp_pose)

					start_frame, end_frame = start_stop(landmarks, mp_pose,
					                                    cap, start_frame,
					                                    end_frame, threshold)

					# Crop and store hip angle data within the squat range
					if start_frame != 0 and end_frame == 0:
						hip_angle_data["L"].append(hip_angle[0])
						hip_angle_data["R"].append(hip_angle[1])

						knee_angle_data["L"].append(knee_angle[0])
						knee_angle_data["R"].append(knee_angle[1])

						ankle_angle_data["L"].append(ankle_angle[0])
						ankle_angle_data["R"].append(ankle_angle[1])

						deviation_angle_data.append(deviation_angle)

						if hip_angle[0] < hip_angle_min:
							# TODO: Refine This algorithmically to avoid
							#  extra compute
							hip_angle_min = hip_angle[0]
							bottom_frame = frame
							angle_femur_left, angle_femur_right = get_deepfemur_angle(
								landmarks,
								mp_pose)

					# TODO: Consider making this plot on screen again ie: if reccording of poses is saved
					# # Visualize angle of hip_angle at the hip
					# cv2.putText(image, str(hip_angle),
					#             tuple(
					# 	            np.multiply(hip, [640, 480]).astype(int)),
					#             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2,
					#             cv2.LINE_AA)

				except Exception as e:
					traceback.print_exc()
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

		make_plot(hip_angle_data, "Hip")
		make_plot(knee_angle_data, "Knee")
		make_plot(ankle_angle_data, "Ankle")
		make_plot(deviation_angle_data, "Deviation")



		bottom_frame = cv2.cvtColor(bottom_frame, cv2.COLOR_BGR2RGB)
		make_handout(angle_femur_left, angle_femur_right, bottom_frame)


def make_plot(angle_data, name):
	plt.figure()
	if (type(angle_data) is dict):
		plt.plot(angle_data["L"])
		plt.plot(angle_data["R"])
	else:
		plt.plot(angle_data)
	plt.xlabel('Frame')
	plt.ylabel(name + ' Angle (degrees)')
	plt.title(name + ' Angle during Squat')
	plt.legend(["Left", "Right"])
	plt.savefig(name + "_angle.png")
	plt.show()
	plt.close()


def make_handout(fl, fr, bottom_frame):
	pdf_path = "functional.pdf"

	# Create a new PDF instance

	pdf = FPDF()

	# Add a new page
	pdf.add_page()

	# Set font and size for the title
	pdf.set_font("Arial", "B", 16)
	pdf.cell(0, 10, "Functional Movement Assessment", 0, 1, "C")


	# Set font and size for the content
	pdf.set_font("Arial", "", 12)

	# Add text content
	pdf.cell(0, 10, "The Functional Movement Assessment criteria is "
	                "visualized below for the Overhead Squat Test ", ln=True)

	# Set font and size for the content
	pdf.set_font("Arial", "", 10)

	# Create a PIL Image object from the CV2 frame
	image_pil = Image.fromarray(bottom_frame)

	# Save the PIL image as a temporary file (optional)
	temp_filename = 'temp_image.jpg'
	image_pil.save(temp_filename)

	# Add the image to the PDF
	pdf.image(temp_filename, 147, 45, 43, 80)
	# Add text conten t
	pdf.cell(0, 20, "Functional Criteria", ln=True)

	file_names_header = ["deviation"]

	# Save the plot as an image
	for i, name in enumerate(file_names_header):
		plot_image_path = name + "_angle.png"

		 # Add the plot image to the PDF
		pdf.image(plot_image_path, x=(10 + (90 * (i % 2))),
		          y=(45),
		          w=100,
		          h=75)

	# Convert the float values to strings with 2 decimal places
	fl_str = "{:.2f}".format(fl)
	fr_str = "{:.2f}".format(fr)

	# Add the text content with centered alignment
	pdf.set_xy((pdf.w-4)/2, 70)
	pdf.cell(0, 10, "Deepest Femur L: " + fl_str, ln=True)

	pdf.set_xy((pdf.w-4)/2, 80)
	pdf.cell(0, 10, "Deepest Femur R: " + fr_str, ln=True)

	# Add text content
	pdf.set_xy(10, 120)
	pdf.cell(0, 10, "Joint Kinematics", ln=True)


	file_names = ["hip", "knee", "ankle"]

	# Save the plot as an image
	for i, name in enumerate(file_names):
		plot_image_path = name + "_angle.png"

		# Add the plot image to the PDF
		pdf.image(plot_image_path, x=(10 + (90 * (i % 2))),
		                           y=(130 + (70 * np.floor(i/2))),
		                           w=100,
		                           h=75)

	# Save the PDF file
	pdf.output(pdf_path)

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


if __name__ == '__main__':
	generate_frames_file()
