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
from analyzer import AnalyzeSquat
import os
import pandas as pd
import csv

#TODO:
# Auto Body Shading

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


def get_filenames():
	import os

	directory = 'data/'
	subdirectories = [os.path.join(directory, subdir) for subdir in os.listdir(directory) if os.path.isdir(os.path.join(directory, subdir))]
	file_list = []

	for subdir in subdirectories:
		if os.path.exists(subdir) and os.path.isdir(subdir):
			files = sorted(os.listdir(subdir))
			file_list.extend([(files[i], files[i + 1], subdir) for i in
			                  range(0, len(files), 2)])

	return file_list


def generate_frames_file():
	'''
	Creates Pose Estimates for a Single Video File with output handout
	'''

	file_list = get_filenames()
	print(file_list)

	for filenames in file_list:

		filename_side = filenames[0]
		filename_front = filenames[1]
		qualitative_label = filenames[2]

	# filename_side = 'IMG_7038.mov'
	# filename_front = 'IMG_7039.mov'
	# qualitative_label = 'data'

		cap = file_import(qualitative_label + '/' + filename_side)
		cap2 = file_import(qualitative_label + '/' + filename_front)

		# Initialize MediaPipe Drawing
		mp_drawing = mp.solutions.drawing_utils
		mp_drawing2 = mp.solutions.drawing_utils

		# Initialize MediaPipe Pose model
		mp_pose = mp.solutions.pose
		mp_pose2 = mp.solutions.pose


		# Computed with the Side View
		start_frame = 0
		end_frame = 0
		hip_angle_data = {"L": [], "R": []}
		knee_angle_data = {"L": [], "R": []}
		ankle_angle_data = {"L": [], "R": []}
		shoulder_deviation_angle_data = []
		deviation_angle_data = []
		foot_angle_data = {"L": [], "R": []}

		threshold = 160
		hip_angle_min = threshold



		# Computed with the Front View
		shin_varvalg_angle_data = {"L": [], "R": []}
		foot_inout_data = {"L": [], "R": []}


		global bottom_frame, angle_femur_left, angle_femur_right, torso_min, \
			shoulder_min
		angle_femur_left = None
		angle_femur_right = None
		bottom_frame = None
		torso_min = None
		shoulder_min = None

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
						shoulder_deviation_angle = get_shoulder_angle(landmarks,
						                                          mp_pose)
						foot_angle = get_foot_angle(landmarks, mp_pose)
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

							shoulder_deviation_angle_data.append(shoulder_deviation_angle)

							foot_angle_data["L"].append(foot_angle[0])
							foot_angle_data["R"].append(foot_angle[1])

							if hip_angle[0] < hip_angle_min:
								hip_angle_min = hip_angle[0]

								bottom_frame = frame

								# Draw on Bottom Frame
								mp_drawing.draw_landmarks(bottom_frame,
								                          results.pose_landmarks,
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

								angle_femur_left, angle_femur_right = get_deepfemur_angle(
									landmarks,
									mp_pose)
								torso_min = deviation_angle
								shoulder_min = shoulder_deviation_angle

								squat_depth = np.round((angle_femur_left +
								               angle_femur_right) / 2)

								print(squat_depth)

								# Visualize angle of hip_angle at the hip
								cv2.putText(bottom_frame, "Depth: ",
								            tuple(
								            np.multiply([landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
		                                    landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y],
					                        [1250, 1920]).astype(int)),
								            cv2.FONT_HERSHEY_SIMPLEX, 2, (245, 66, 230), 2,
								            cv2.LINE_AA)

								# Visualize angle of hip_angle at the hip
								cv2.putText(bottom_frame, str(squat_depth),
								            tuple(
									            np.multiply([landmarks[
										                         mp_pose.PoseLandmark.LEFT_HIP.value].x,
									                         landmarks[
										                         mp_pose.PoseLandmark.LEFT_HIP.value].y],
									                        [1250, 2000]).astype(
										            int)),
								            cv2.FONT_HERSHEY_SIMPLEX, 2,
								            (245, 66, 230), 2,
								            cv2.LINE_AA)

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

				except Exception as e:
					traceback.print_exc()
					break


			cap.release()
			cv2.destroyAllWindows()

			make_plot(hip_angle_data, "Hip")
			make_plot(knee_angle_data, "Knee")
			make_plot(ankle_angle_data, "Ankle")
			make_plot(deviation_angle_data, "Deviation")
			make_plot(shoulder_deviation_angle_data, "Shoulder Deviation")
			make_plot(foot_angle_data, "Foot")

			bottom_front, knee_vv_min = front_view(cap2, mp_drawing2, mp_pose2,
			                               start_frame,
			                        end_frame,
			           shin_varvalg_angle_data, foot_inout_data)

			bottom_frame = cv2.cvtColor(bottom_frame, cv2.COLOR_BGR2RGB)
			bottom_front = cv2.cvtColor(bottom_front, cv2.COLOR_BGR2RGB)

			make_handout(angle_femur_left, angle_femur_right, bottom_frame)

			print(torso_min)
			print(shoulder_min)


			analyze_data(hip_angle_data, knee_angle_data, ankle_angle_data,
			             deviation_angle_data, shoulder_deviation_angle_data,
			             torso_min, shoulder_min,
			             shin_varvalg_angle_data,
			             (angle_femur_left, angle_femur_right),
			             knee_vv_min, bottom_frame, bottom_front, filename_side,
			             filename_front, foot_angle_data, foot_inout_data,
			             qualitative_label)


def front_view(cap2, mp_drawing2, mp_pose2, start_frame, end_frame,
               shin_varvalg_angle_data, foot_inout_data):

	threshold = 160
	hip_angle_min = threshold
	knee_vv_min = threshold

	with mp_pose2.Pose(min_detection_confidence=0.5,
	                  min_tracking_confidence=0.5) as pose:
		while cap2.isOpened():
			try:
				retFr, frameFr = cap2.read()

				# Recolor image to RGB
				imageFr = cv2.cvtColor(frameFr, cv2.COLOR_BGR2RGB)
				imageFr.flags.writeable = False

				# Make detection
				resultsFr = pose.process(imageFr)

				# Recolor back to BGR
				imageFr.flags.writeable = True
				imageFr = cv2.cvtColor(imageFr, cv2.COLOR_RGB2BGR)

				frame_num = cap2.get(cv2.CAP_PROP_POS_FRAMES)

				try:
					landmarksFr = resultsFr.pose_landmarks.landmark

					# Calculate angles
					shin_varvalg_angle = get_varvalg_angle(landmarksFr,
					                                            mp_pose2)

					foot_inout = get_inout_angle(landmarksFr, mp_pose2)

					hip_angle = get_hip_angle(landmarksFr, mp_pose2)

					if hip_angle[0] < hip_angle_min:
						hip_angle_min = hip_angle[0]
						bottom_front = frameFr

						# Draw on Bottom Frame
						mp_drawing2.draw_landmarks(bottom_front,
						                          resultsFr.pose_landmarks,
						                          mp_pose2.POSE_CONNECTIONS,
						                          mp_drawing2.DrawingSpec(
							                          color=(245, 117, 66),
							                          thickness=2,
							                          circle_radius=2),
						                          mp_drawing2.DrawingSpec(
							                          color=(245, 66, 230),
							                          thickness=2,
							                          circle_radius=2)
						                          )

						knee_vv_min = shin_varvalg_angle

					# Crop and store hip angle data within the squat range
					if frame_num >= start_frame and frame_num <= end_frame:
						shin_varvalg_angle_data["L"].append(
							-1 * shin_varvalg_angle[0])
						shin_varvalg_angle_data["R"].append(
							shin_varvalg_angle[1])

						foot_inout_data["L"].append(foot_inout[0])
						foot_inout_data["R"].append(foot_inout[1])

				except Exception as e:
					traceback.print_exc()
					pass

				# Render detections
				mp_drawing2.draw_landmarks(imageFr, resultsFr.pose_landmarks,
				                          mp_pose2.POSE_CONNECTIONS,
				                          mp_drawing2.DrawingSpec(
					                          color=(245, 117, 66),
					                          thickness=2,
					                          circle_radius=2),
				                          mp_drawing2.DrawingSpec(
					                          color=(245, 66, 230),
					                          thickness=2,
					                          circle_radius=2)
				                          )

				# Display the image
				cv2.imshow('Squat Video Front', imageFr)

				if cv2.waitKey(1) & 0xFF == ord('q'):
					break

			except Exception as e:
				traceback.print_exc()
				break

		cap2.release()
		cv2.destroyAllWindows()

		make_plot(shin_varvalg_angle_data, "VarValg")
		make_plot(foot_inout_data, "Foot Rotation")

		return bottom_front, knee_vv_min


def analyze_data(hip_angle, knee_angle, ankle_angle, dev_angle, dev_shoulder,
			torso_min, shoulder_min,
			vv_angle, deepest, knee_vv_min, bottom_frame, bottom_front,
			filename_side, filename_front, foot_angle, inout_angle, label):

	analyzer = AnalyzeSquat(hip_angle, knee_angle, ankle_angle, dev_angle,
			dev_shoulder, torso_min, shoulder_min, vv_angle, deepest,
			knee_vv_min, bottom_frame, bottom_front,
			filename_side,
			filename_front, foot_angle, inout_angle, label)

	# Check for ailments
	analyzer.test()
	analyzer.make_profile(filename_side.split(".")[0])



def make_plot(angle_data, name):
	plt.figure()
	if (type(angle_data) is dict):
		plt.plot(angle_data["L"])
		plt.plot(angle_data["R"])
		plt.legend(["Left", "Right"])
	else:
		plt.plot(angle_data)
	plt.xlabel('Frame')
	plt.ylabel(name + ' Angle (degrees)')
	plt.title(name + ' Angle during Squat')
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


	file_names = ["hip", "knee", "ankle", "varvalg"]

	# Save the plot as an image
	for i, name in enumerate(file_names):
		plot_image_path = name + "_angle.png"

		# Add the plot image to the PDF
		pdf.image(plot_image_path, x=(10 + (90 * (i % 2))),
		                           y=(130 + (70 * np.floor(i/2))),
		                           w=100,
		                           h=75)

	pdf.add_page()

	file_names = ["shoulder deviation", "foot", "foot rotation"]

	# Save the plot as an image
	for i, name in enumerate(file_names):
		plot_image_path = name + "_angle.png"

		# Add the plot image to the PDF
		pdf.image(plot_image_path, x=(10 + (90 * (i % 2))),
		          y=(10 + (70 * np.floor(i / 2))),
		          w=100,
		          h=75)

	# Save the PDF file
	pdf.output(pdf_path)

if __name__ == '__main__':
	generate_frames_file()

