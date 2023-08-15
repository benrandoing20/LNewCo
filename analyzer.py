# Created to take in a single subjects squat profile and indicate which of
# the potential ailments they exhibit then querying recommendations

# Created by Ben Randoing during 07/2023

import numpy as np
import matplotlib.pyplot as plt
from fpdf import FPDF
from PIL import Image
from scipy.integrate import trapz
import pdfkit
import os
import pandas as pd

class AnalyzeSquat():
	######################################################################
	######################## Init and Variable Declare ###################

	def __init__(self, hip, knee, ankle, dev, devs, torso, shoulder, vv, deep,
	             vv_min, bottom_frame, base_frame, file_side, file_font,
	             foot, inout, label):
		self.side = file_side
		self.front = file_font
		self.label = label
		self.hip = hip
		self.knee = knee
		self.ankle = ankle
		self.dev = dev
		self.devs = devs
		self.torso_min = [torso]
		self.sho_min = [shoulder]
		self.deep = deep
		self.vv = vv
		self.vv_min = vv_min
		self.foot = foot
		self.inout = inout
		self.squat_profile = []
		self.ailments = []
		self.interventions = {}
		self.bottom_frame = bottom_frame
		self.base_frame = base_frame
		self.asym_max = 5000
		self.hip_data = {"auc": [], "conv": []}
		self.knee_data = {"auc": [], "conv": []}
		self.assymetric_score = {"hip_angle": 0.0, "knee_angle": 0.0,
		                         "ankle_angle": 0.0, "deep_diff": 0.0,
		                         "vert_offset": 0.0}
		self.knee_score = {"vv_score": 0.0}
		self.core_score = {"deviation": 0.0, "arms": 0.0}
		self.squat_score = {"deep_femur": 0.0}
		self.final_scores = {"asymmetric_score": 0, "knee_stability_score":
							 0, "core_strength_score": 0, "squat_score": 0}
		self.output_class = {"vv": "None", "foot_out": "No",
		                     "heel_raise": "No", "shift": "No",
		                     "lean": "No", "arms_forward": "No"}


		self.ailments_store = {
			"not_deep_l": ["Did not reach a squat depth below horizontal L",
			               "not_deep_l"],
			"not_deep_r": ["Did not reach a squat depth below horizontal R",
			               "not_deep_r"],
			"depth_aym": ["Demonstrated asymmetry at squat base",
			              "We recommend rolling out the weak side adductor "
			              "and IT-Band as well as the strong side Biceps "
			              "Femoris, Gastrocnemius, Soleus, and Piriformis. "
			              "Please also consider strengthening exercises "
			              "including weak side leg pulls and strong side "
			              "leg pushes."],
			"forward_lean": ["Forward lean of trunk", "We recommend rolling "
			                                          "out"
			                                          "the gastrocnemius, "
			                                          "soleus, and hip "
			                                          "flexor. Please also "
			                                          "consider "
			                                          "strengthening with "
			                                          "floor cobras."],
			"Varus": ["Demonstrated excess knee varus", "We recommend rolling "
			                                          "out"
			                                          "the gastrocnemius, "
			                                          "soleus, and Biceps "
			                                          "Femorus."],
			"Valgus": ["Demonstrated excess knee valgus", "We recommend rolling "
			                                          "out"
			                                          "the gastrocnemius, "
			                                          "soleus, "
			                                          "and Adductors."]}
		self.prediction_roll_count = {"CalfL": 0,
		                              "CalfR": 0,
		                              "Quads": 0,
		                              "Glutes": 0,
		                              "AdductorsL": 0,
		                              "AdductorsL": 0,
		                              "Thoracic Spine": 0,
		                              "Shoulder": 0}


	######################################################################
	######################## Asymmetry Feature Creation ##################

	def check_deep(self):
		# -10 to 10 is Green
		# -15 TO 15  multiply by 2
		# More outside multiply by 3

		left = self.deep[0]
		right = self.deep[1]
		diff = right - left

		if diff > -10 or diff < 10:
			self.assymetric_score["deep_diff"] = 50 + 0.8 * diff
		else:
			self.assymetric_score["deep_diff"] = 50 + diff


		### Squat Score Contribution ###
		mean = (left + right) / 2
		print('mean: ' + str(mean))
		if mean < 0:
			self.squat_score["deep_femur"] = 100
		elif mean < 10:
			self.squat_score["deep_femur"] = 100 - 2 * mean
		else:
			self.squat_score["deep_femur"] = 100 - 3 * mean

		### Score for Rolling Predictions ###
		# Negative mean indicates there is a rightward lean and strong side
		if mean < 0:
			self.prediction_roll_count["AdductorsL"] += 1
			self.prediction_roll_count["CalfR"] += 1




	def check_hip(self):
		hip_data = self.hip
		right_hip_area = trapz(hip_data["R"])
		left_hip_area = trapz(hip_data["L"])

		hip_right_norm = np.array(hip_data["R"] / trapz(hip_data["R"]))
		hip_left_norm = np.array(hip_data["L"] / trapz(hip_data["L"]))

		hip_auc = (right_hip_area - left_hip_area) / ((right_hip_area
		                                               + left_hip_area) / 2)


		hip_conv = np.convolve(hip_right_norm, hip_left_norm, mode='full')

		hip_conv_max = np.max(hip_conv)
		hip_conv_min = np.min(hip_conv)
		hip_conv_asym = 1 - (hip_conv_max - hip_conv_min) / 0.02
		print(hip_conv_asym)

		self.hip_data["auc"] = hip_auc
		self.hip_data["conv"] = list(hip_conv)

		hip_asymmetry = (0.6 * hip_conv_asym + 0.4 * np.abs(hip_auc)) * \
		                np.sign(hip_auc)

		print(hip_asymmetry)

		self.assymetric_score["hip_angle"] = 50 + hip_asymmetry * 50
		print(self.assymetric_score["hip_angle"])

	def check_knee(self):
		knee_data = self.knee
		right_knee_area = trapz(knee_data["R"])
		left_knee_area = trapz(knee_data["L"])

		knee_right_norm = np.array(knee_data["R"] / trapz(knee_data["R"]))
		knee_left_norm = np.array(knee_data["L"] / trapz(knee_data["L"]))

		knee_auc = (right_knee_area - left_knee_area) / ((right_knee_area
		                                                  + left_knee_area)
		                                                 / 2)
		print(knee_auc)

		knee_conv = np.convolve(knee_right_norm, knee_left_norm, mode='full')


		knee_conv_max = np.max(knee_conv)
		knee_conv_min = np.min(knee_conv)
		knee_conv_asym = 1 - (knee_conv_max - knee_conv_min) / 0.02
		print(knee_conv_asym)

		self.knee_data["auc"] = knee_auc
		self.knee_data["conv"] = list(knee_conv)

		knee_asymmetry = (0.6 * knee_conv_asym + 0.4 * np.abs(knee_auc)) * \
		                np.sign(knee_auc)

		print(knee_asymmetry)

		self.assymetric_score["knee_angle"] = 50 + knee_asymmetry * 50

		print(self.assymetric_score["knee_angle"])

	def check_ankle(self):
		ankle_data = self.ankle
		right_ankle_area = trapz(ankle_data["R"])
		left_ankle_area = trapz(ankle_data["L"])

		ankle_right_norm = np.array(ankle_data["R"] / trapz(ankle_data["R"]))
		ankle_left_norm = np.array(ankle_data["L"] / trapz(ankle_data["L"]))

		ankle_auc = (right_ankle_area - left_ankle_area) / ((right_ankle_area
		                                                  + left_ankle_area) / 2)
		print(ankle_auc)

		ankle_conv = np.convolve(ankle_right_norm, ankle_left_norm,
		                         mode='full')

		ankle_conv_max = np.max(ankle_conv)
		ankle_conv_min = np.min(ankle_conv)
		ankle_conv_asym = 1 - (ankle_conv_max - ankle_conv_min) / 0.02
		print(ankle_conv_asym)

		ankle_asymmetry = (0.6 * ankle_conv_asym + 0.4 * np.abs(ankle_auc)) * \
		                 np.sign(ankle_auc)

		print(ankle_asymmetry)
		self.assymetric_score["ankle_angle"] = 50 + ankle_asymmetry * 50

		print(self.assymetric_score["ankle_angle"])

	# def check_Asym(self):
	# 	# TODO: Hip y coord Asymmetry Value -> Bin -> Save in Asymmetry Outputs
	#
	######################## Knee Stability Creation ##################
	def check_VarValg(self):
		knee_bend = self.vv
		vv_min_sum = np.abs(self.vv_min[0]) + np.abs(self.vv_min[1])
		sign = self.vv_min[1] - self.vv_min[0] # R - L

		if vv_min_sum < 5:
			self.knee_score["vv_score"] = (sign * 0.2 * vv_min_sum) / 2 + 50
		elif vv_min_sum < 20:
			self.knee_score["vv_score"] = (sign * 0.4 * vv_min_sum) / 2 + 50
		elif vv_min_sum < 40:
			self.knee_score["vv_score"] = (sign * 0.6 * vv_min_sum) / 2 + 50
		else:
			self.knee_score["vv_score"] = (sign * vv_min_sum) / 2 + 50


	######################## Core Strength Creation ##################
	def check_ForDev(self):
		dev_min = abs(self.torso_min)
		self.core_score["deviation"] = dev_min
		print(self.core_score["deviation"])


	def check_ArmsFor(self):
		devs_min = abs(self.sho_min)
		self.core_score["arms"] = devs_min
		print(self.core_score["arms"])


	###################################################################
	######################## Aggregate Sub scores #####################

	def agg_scores(self):
		self.check_deep()
		self.check_hip()
		self.check_knee()
		self.check_ankle()
		self.check_VarValg()
		self.check_ForDev()
		self.check_ArmsFor()

		# Asymmetry Score
		for key, value in self.assymetric_score.items():
			if key == "deep_diff":
				self.final_scores["asymmetric_score"] += value

		# Knee Stability Score
		for key, value in self.knee_score.items():
			self.final_scores["knee_stability_score"] += value

		if self.final_scores["knee_stability_score"] < 0 or \
		   self.final_scores["knee_stability_score"] > 100:
			self.final_scores["knee_stability_score"] = 0

		# Core Stability Score
		for key, value in self.core_score.items():
			self.final_scores["core_strength_score"] += value
		self.final_scores["core_strength_score"] /= len(self.core_score)
		print(self.final_scores["core_strength_score"])

		# Squat Score
		for key, value in self.squat_score.items():
			self.final_scores["squat_score"] += value
		self.final_scores["squat_score"] /= len(self.squat_score)
		if self.final_scores["squat_score"] < 0:
			self.final_scores["squat_score"] = 0


	###################################################################
	######################## Front Facing Function ####################

	def test(self):
		self.create_dataset()
		self.create_bullet_list()
		self.create_recs_dictionary()
		self.agg_scores()
		self.create_gauge_chart(self.final_scores[
			"squat_score"], 100, 33, 67, 'Squat.png')
		self.create_gauge_chart(self.final_scores[
			"core_strength_score"], 100, 33, 67, 'Core.png')
		self.create_gauge_chart_sym(self.final_scores[
			"knee_stability_score"], 100, 20, 40, 'KneeStability.png')
		self.create_gauge_chart_sym(self.final_scores[
			"asymmetric_score"], 100, 20, 40, 'Asymmetry.png')
		self.add_row()



	###################################################################
	######################## Create Output Squat Profile ##############

	def create_gauge_chart(self, current_value, max_value, red_value,
	                       green_value, filename):
		# Create figure and axis
		fig, ax = plt.subplots(figsize=(8, 2))

		# Define colors
		red_color = '#FE4444'
		yellow_color = '#FEF344'
		green_color = '#00DD5A'

		# Set the range for the gauge chart
		ax.set_xlim(0, max_value)
		ax.set_ylim(0, 0.8)

		# Create the shaded red, yellow, and green regions
		ax.axvspan(0, red_value, facecolor=red_color, alpha=0.5)
		ax.axvspan(red_value, green_value, facecolor=yellow_color, alpha=0.5)
		ax.axvspan(green_value, 100, facecolor=green_color, alpha=0.5)

		# Create the tick indicating the current position
		ax.axvline(x=current_value, color='white', linewidth=10, clip_on=False)

		# Set the x-axis and y-axis labels and their colors
		ax.set_xlabel('X-axis', color='white')
		ax.set_ylabel('Y-axis', color='white')

		# Hide the y-axis and ticks
		ax.get_yaxis().set_visible(False)
		ax.get_xaxis().set_visible(False)


		# Save the figure with a transparent background
		plt.savefig(filename, transparent=True)

		# Show the plot
		plt.show()

	def create_gauge_chart_sym(self, current_value, max_value, red_value,
	                       green_value, filename):
		# Create figure and axis
		fig, ax = plt.subplots(figsize=(8, 2))

		# Define colors
		red_color = '#FE4444'
		yellow_color = '#FEF344'
		green_color = '#00DD5A'

		# Set the range for the gauge chart
		ax.set_xlim(0, max_value)
		ax.set_ylim(0, 0.8)

		# Create the shaded red, yellow, and green regions
		ax.axvspan(0, red_value, facecolor=red_color, alpha=0.5)
		ax.axvspan(100 - red_value, 100, facecolor=red_color, alpha=0.5)
		ax.axvspan(red_value, green_value, facecolor=yellow_color, alpha=0.5)
		ax.axvspan(100 - green_value, 100 - red_value, facecolor=yellow_color,
		           alpha=0.5)
		ax.axvspan(green_value, 100 - green_value, facecolor=green_color,
		           alpha=0.5)

		# Create the tick indicating the current position
		ax.axvline(x=current_value, color='white', linewidth=10, clip_on=False)

		# Set the x-axis and y-axis labels and their colors
		ax.set_xlabel('X-axis', color='white')
		ax.set_ylabel('Y-axis', color='white')

		# Hide the y-axis and ticks
		ax.get_yaxis().set_visible(False)
		ax.get_xaxis().set_visible(False)

		# Save the figure with a transparent background
		plt.savefig(filename, transparent=True)

		# Show the plot
		plt.show()

	def create_bullet_list(self):
		for i, val in enumerate(self.squat_profile):
			self.ailments.append(self.ailments_store[val][0])

	def create_recs_dictionary(self):
		for i, val in enumerate(self.squat_profile):
			self.interventions[self.ailments_store[val][0]] = (
				self.ailments_store[val][1])

	def bullet_point_list(self, items, pdf):
		pdf.set_font("Arial", "B", 10)
		for item in items:
			pdf.cell(0, 10, "  - " + item, ln=True)

	def headed_bullet_point_list(self, items_dict, pdf):
		pdf.set_font("Arial", "B", 10)
		print(items_dict)
		for key, value in items_dict.items():
			pdf.cell(0, 10, "" + key, ln=True)
			pdf.cell(0, 10, "  - " + value, ln=True)

	def draw_rounded_rect(self, pdf, x, y, w, h, r):
		pdf.ellipse(x, y, 2 * r, 2 * r, 'F')  # Top-left corner
		pdf.ellipse(x + w - 2 * r, y, 2 * r, 2 * r, 'F')  # Top-right corner
		pdf.ellipse(x, y + h - 2 * r, 2 * r, 2 * r, 'F')  # Bottom-left
		            # corner
		pdf.ellipse(x + w - 2 * r, y + h - 2 * r, 2 * r, 2 * r, 'F')  #
		# Bottom-right
		# corner

		pdf.rect(x + r, y, w - 2 * r, r + 5, 'F')  # Top side
		pdf.rect(x, y + r, w, h - 2 * r, 'F')  # Middle side (center part)
		pdf.rect(x + r, y + h - r - 5, w - 2 * r, r + 5, 'F')  # Bottom side

	def make_profile(self):
		'''
		Function to Generate PDF Report of Raw Graphs for Internal Use
		'''
		pdf_path = "profile.pdf"

		# Create a new PDF instance
		pdf = FPDF()
		pdf.add_page()

		pdf.set_text_color(255, 255, 255)
		pdf.set_fill_color(48, 49, 57)
		pdf.rect(0, 0, pdf.w, pdf.h, 'F')  # 'F' means fill the rectangle

		pdf.set_fill_color(23, 24, 32)

		self.draw_rounded_rect(pdf, 8, 36, 194, 84, 10)
		self.draw_rounded_rect(pdf, 8, 132, 94, 154, 10)
		self.draw_rounded_rect(pdf, 108, 132, 94, 154, 10)
		# self.draw_rounded_rect(pdf, 108, 200, 94, 86, 10)



		# RGB color
		pdf.set_fill_color(48, 49, 57)
		# Rounded rectangle with 10mm radius for corners
		self.draw_rounded_rect(pdf, 12, 40, 90, 36, 10)
		self.draw_rounded_rect(pdf, 108, 40, 90, 36, 10)
		self.draw_rounded_rect(pdf, 12, 80, 90, 36, 10)
		self.draw_rounded_rect(pdf, 108, 80, 90, 36, 10)

		self.draw_rounded_rect(pdf, 12, 136, 86, 146, 10)
		self.draw_rounded_rect(pdf, 112, 136, 86, 146, 10)



		# Set font and size for the title
		pdf.set_font("Arial", "B", 20)
		pdf.cell(0, 10, "Overhead Squat FMS", 0, 1, "C")

		pdf.set_font("Arial", "", 12)

		# Create a PIL Image object from the CV2 frame
		image_pil = Image.fromarray(self.bottom_frame)
		image_pil_front = Image.fromarray(self.base_frame)

		# Save the PIL image as a temporary file
		temp_filename = 'temp_image.jpg'
		temp_filename_front = 'temp_image_front.jpg'

		image_pil.save(temp_filename)
		image_pil_front.save(temp_filename_front)

		pdf.image(temp_filename,  20, 147, 70)
		pdf.image(temp_filename_front,  120, 147, 70)

		pdf.image('Squat.png', 7, 60, 96, 10)
		pdf.image('Core.png', 104, 60, 96, 10)
		pdf.image('KneeStability.png', 7, 100, 96, 10)
		pdf.image('Asymmetry.png', 104, 100, 96, 10)

		# Add text content
		pdf.set_font("Arial", "B", 14)
		pdf.cell(0, 20, "Performance Evaluation", ln=True)

		#Add Sub_labels
		pdf.set_font("Arial", "B", 12)
		pdf.set_xy(36, 38)
		pdf.cell(0, 20, "Squat Score: {:.0f}".format(self.final_scores[
			                                              'squat_score']), ln=True)
		pdf.set_xy(128, 78)
		pdf.cell(0, 20, "Asymmetry Score: {:.0f}".format(self.final_scores['asymmetric_score']), ln=True)
		pdf.set_xy(34, 78)
		pdf.cell(0, 20, "Knee Stability : {:.0f}".format(self.final_scores['knee_stability_score']), ln=True)
		pdf.set_xy(132, 38)
		pdf.cell(0, 20, "Core Strength: {:.0f}".format(self.final_scores['core_strength_score']), ln=True)



		# Add text content
		pdf.set_xy(10, 122)
		pdf.set_font("Arial", "B", 14)
		pdf.cell(0, 10, "Recommended Interventions", ln=True)


		roll_rec = self.get_max_dict(self.prediction_roll_count)

		# # Roll
		# pdf.set_xy(16, 142)
		# pdf.set_font("Arial", "B", 12)
		# pdf.cell(0, 10, roll_rec, ln=True)
		#
		# # Strengthen
		# pdf.set_xy(16, 162)
		# pdf.set_font("Arial", "B", 12)
		# pdf.cell(0, 10, "2. Test", ln=True)

		# Page 2
		pdf.add_page()

		pdf.set_text_color(255, 255, 255)
		pdf.set_fill_color(48, 49, 57)
		pdf.rect(0, 0, pdf.w, pdf.h, 'F')  # 'F' means fill the rectangle

		pdf.set_fill_color(23, 24, 32)

		self.draw_rounded_rect(pdf, 8, 36, 194, 250, 10)

		# Set font and size for the title
		pdf.set_font("Arial", "B", 20)
		pdf.cell(0, 10, "Muscle Target Regions", 0, 1, "C")

		# Add text content
		pdf.set_font("Arial", "B", 14)
		pdf.cell(0, 20, "Darker shades of red indicate regions for potential muscle strengthening", ln=True)

		pdf.image('bodymap-master/molemapper-randheat.png', 15, 75, 180, 170)

		# Page 3
		pdf.add_page()

		pdf.set_text_color(255, 255, 255)
		pdf.set_fill_color(48, 49, 57)
		pdf.rect(0, 0, pdf.w, pdf.h, 'F')  # 'F' means fill the rectangle

		pdf.set_fill_color(23, 24, 32)

		self.draw_rounded_rect(pdf, 8, 36, 194, 246, 10)

		pdf.set_fill_color(48, 49, 57)
		self.draw_rounded_rect(pdf, 12, 40, 90, 56, 10)
		self.draw_rounded_rect(pdf, 108, 40, 90, 56, 10)
		self.draw_rounded_rect(pdf, 12, 100, 90, 56, 10)
		self.draw_rounded_rect(pdf, 108, 100, 90, 56, 10)

		self.draw_rounded_rect(pdf, 12, 160, 90, 56, 10)
		self.draw_rounded_rect(pdf, 108, 160, 90, 56, 10)
		self.draw_rounded_rect(pdf, 12, 220, 90, 56, 10)
		self.draw_rounded_rect(pdf, 108, 220, 90, 56, 10)


		# Set font and size for the title
		pdf.set_font("Arial", "B", 20)
		pdf.cell(0, 10, "FMS Classification", 0, 1, "C")

		# self.output_class = {"vv": "None", "foot_out": "No",
		#                      "heel_raise": "No", "shift": "No",
		#                      "lean": "No", "arms_forward": "No"}

		# Add Sub_labels
		pdf.set_font("Arial", "B", 18)
		pdf.set_xy(24, 38)
		pdf.cell(0, 20, "Knee Varus or Valgus", ln=True)
		pdf.set_xy(47, 60)
		pdf.cell(0, 20, self.output_class["vv"], ln=True)

		pdf.set_xy(126, 98)
		pdf.cell(0, 20, "Asymmetric Shift", ln=True)
		pdf.set_xy(146, 120)
		pdf.cell(0, 20, self.output_class["shift"], ln=True)

		pdf.set_xy(38, 98)
		pdf.cell(0, 20, "Heel Raise", ln=True)
		pdf.set_xy(50, 120)
		pdf.cell(0, 20, self.output_class["heel_raise"], ln=True)

		pdf.set_xy(130, 38)
		pdf.cell(0, 20, "Foot Turn Out", ln=True)
		pdf.set_xy(146, 60)
		pdf.cell(0, 20, self.output_class["foot_out"], ln=True)

		pdf.set_xy(34, 158)
		pdf.cell(0, 20, "Forward Lean", ln=True)
		pdf.set_xy(50, 180)
		pdf.cell(0, 20, self.output_class["lean"], ln=True)

		pdf.set_xy(125, 158)
		pdf.cell(0, 20, "Arms Fall Forward", ln=True)
		pdf.set_xy(146, 180)
		pdf.cell(0, 20, self.output_class["arms_forward"], ln=True)

		pdf.set_xy(34, 218)
		pdf.cell(0, 20, "", ln=True)

		pdf.set_xy(128, 218)
		pdf.cell(0, 20, "", ln=True)

		# Save the PDF file
		pdf.output(pdf_path)

	def get_max_dict(self, dictionary):
		max_value = -1
		max_key = None
		for key, value in dictionary.items():
			if value > max_value:
				max_value = value
				max_key = key
		return max_key

	def create_dataset(self):
		directory = 'data/'
		csv_filename = 'dataset.csv'

		csv_path = os.path.join(directory, csv_filename)

		if os.path.exists(csv_path):
			print(f"The CSV file '{csv_filename}' already exists.")
		else:
			# Create the CSV file
			empty_df = pd.DataFrame(columns=['Side Filename', 'Front Filename',
			                                 'Frames', 'Hip Angle',
			                                 'Knee Angle',
			                                 'Ankle Angle', 'Deviation Angle',
			                                 'Shoulder Deviation Angle',
			                                 'Torso Min',
			                                 'Shoulder Min',
			                                 'Foot Angle', 'VarValg Angle',
			                                 'VV Min',
			                                 'Foot Inout Angle', 'Deep Femur',
			                                 'Asymmetric Score',
			                                 'Knee Stability',
			                                 'Core Strength',
			                                 'Hip Data',
			                                 'Knee Data',
			                                 'Qualitative Quality'])
			empty_df.to_csv(csv_path, index=False)
			print(f"Empty CSV file '{csv_filename}' has been created.")

	def add_row(self):
		print(self.torso_min, self.sho_min)
		new_row_data = {'Side Filename': self.side,
		                'Front Filename': self.front,
		                'Frames': len(self.hip),
		                'Hip Angle': [[self.hip]],
		                'Knee Angle': [[self.knee]],
		                'Ankle Angle': [[self.ankle]],
		                'Deviation Angle': [[self.dev]],
		                'Shoulder Deviation Angle': [[self.devs]],
		                'Torso Min': [[self.torso_min]],
		                'Shoulder Min': [[self.sho_min]],
		                'Foot Angle': [[self.foot]],
		                'VarValg Angle': [[self.vv]],
		                'VV Min': [[self.vv_min]],
		                'Foot Inout Angle': [[self.inout]],
		                'Deep Femur': [self.deep],
		                'Asymmetric Score': [self.assymetric_score],
		                'Knee Stability': [self.knee_score],
		                'Core Strength': [self.core_score],
		                'Hip Data': [self.hip_data],
		                'Knee Data': [self.knee_data],
		                'Qualitative Quality': self.label}

		new_row_df = pd.DataFrame(new_row_data, index=[0])

		csv_path = 'data/dataset.csv'
		existing_df = pd.read_csv(csv_path)
		side_filenames = existing_df['Side Filename'].tolist()

		print(existing_df)

		if not self.side in side_filenames:
			updated_data = pd.concat([existing_df, new_row_df],
			                         ignore_index=True)
			updated_data.to_csv(csv_path, index=False)
			print("New row appended to the CSV.")
		else:
			print("Row already exists in the CSV.")