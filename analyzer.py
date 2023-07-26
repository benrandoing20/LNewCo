# Created to take in a single subjects squat profile and indicate whicch of
# the potential ailments they exhibit then querying recommendations
# Created by Ben Randoing during 07/2023

import numpy as np
import matplotlib.pyplot as plt
from fpdf import FPDF
from PIL import Image
from scipy.integrate import trapz
import pdfkit



class AnalyzeSquat():
	######################################################################
	######################## Init and Variable Declare ###################

	def __init__(self, hip, knee, ankle, dev, vv, deep, bottom_frame, base_frame):
		self.hip = hip
		self.knee = knee
		self.ankle = ankle
		self.dev = dev
		self.deep = deep
		self.vv = vv
		self.squat_profile = []
		self.ailments = []
		self.interventions = {}
		self.bottom_frame = bottom_frame
		self.base_frame = base_frame
		self.assymetric_score = {"hip_angle": 0.0, "knee_angle": 0.0,
		                         "ankle_angle": 0.0, "deep_diff": 0.0,
		                         "vert_offset": 0.0}
		self.final_scores = {"asymmetric_score": 0, "knee_stability_score":
							 0, "core_strength_score": 0, "squat_score": 50}
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


	######################################################################
	######################## Asymmetry Feature Creation ##################

	def check_deep(self):
		left = self.deep[0]
		right = self.deep[1]

		diff = right - left # right positive convention
		print(diff)

		if -5 < diff < 5:
			self.assymetric_score["deep_diff"] = 3
		elif 0 < diff <= 10:
			self.assymetric_score["deep_diff"] = 4
		elif -10 <= diff < 0:
			self.assymetric_score["deep_diff"] = 2
		elif diff > 10:
			self.assymetric_score["deep_diff"] = 5
		elif diff < -10:
			self.assymetric_score["deep_diff"] = 1

	def check_hip(self):
		hip_data = self.hip
		right_hip_area = trapz(hip_data["R"])
		left_hip_area = trapz(hip_data["L"])

		hip_auc_diff = right_hip_area - left_hip_area
		print(hip_auc_diff)

		if -500 < hip_auc_diff < 500:
			self.assymetric_score["hip_angle"] = 3
		elif 0 < hip_auc_diff <= 1000:
			self.assymetric_score["hip_angle"] = 4
		elif -1000 <= hip_auc_diff < 0:
			self.assymetric_score["hip_angle"] = 2
		elif hip_auc_diff > 1000:
			self.assymetric_score["hip_angle"] = 5
		elif hip_auc_diff < -1000:
			self.assymetric_score["hip_angle"] = 1

	def check_knee(self):
		knee_data = self.knee
		right_knee_area = trapz(knee_data["R"])
		left_knee_area = trapz(knee_data["L"])

		knee_auc_diff = right_knee_area - left_knee_area
		print(knee_auc_diff)

		if -500 < knee_auc_diff < 500:
			self.assymetric_score["knee_angle"] = 3
		elif 0 < knee_auc_diff <= 1000:
			self.assymetric_score["knee_angle"] = 4
		elif -1000 <= knee_auc_diff < 0:
			self.assymetric_score["knee_angle"] = 2
		elif knee_auc_diff > 10000:
			self.assymetric_score["knee_angle"] = 5
		elif knee_auc_diff < -10000:
			self.assymetric_score["knee_angle"] = 1

	def check_ankle(self):
		ankle_data = self.ankle
		right_ankle_area = trapz(ankle_data["R"])
		left_ankle_area = trapz(ankle_data["L"])

		ankle_auc_diff = right_ankle_area - left_ankle_area
		print(ankle_auc_diff)

		if -500 < ankle_auc_diff < 500:
			self.assymetric_score["ankle_angle"] = 3
		elif 0 < ankle_auc_diff <= 1000:
			self.assymetric_score["ankle_angle"] = 4
		elif -1000 <= ankle_auc_diff < 0:
			self.assymetric_score["ankle_angle"] = 2
		elif ankle_auc_diff > 1000:
			self.assymetric_score["ankle_angle"] = 5
		elif ankle_auc_diff < -1000:
			self.assymetric_score["ankle_angle"] = 1

	# def check_Asym(self):
	# 	# TODO: Hip y coord Asymmetry Value -> Bin -> Save in Asymmetry Outputs
	#
	# ######################## Knee Stability Creation ##################
	# def check_VarValg(self):
	# 	# TODO: Varus and Valgus Angle check angle Value -> Bin -> Knee Out
	#
	#
	# ######################## Knee Stability Creation ##################
	# def check_ForDev(self):
	# 	# TODO: Forward Deviation Angle check angle Asymmetry Value -> Bin ->
	#
	# def check_ArmsFor(self):
	# 	# TODO: Forward Arms Angle check angle Asymmetry Value -> Bin ->

	###################################################################
	######################## Aggregate Sub scores #####################

	def agg_scores(self):
		self.check_deep()
		self.check_hip()
		self.check_knee()
		self.check_ankle()

		asym_max = 20
		knee = 0
		core = 0
		for key, value in self.assymetric_score.items():
			self.final_scores["asymmetric_score"] += value
		self.final_scores["asymmetric_score"] /= asym_max
		self.final_scores["asymmetric_score"] *= 100




	###################################################################
	######################## Front Facing Function ####################

	def test(self):
		self.create_bullet_list()
		self.create_recs_dictionary()
		self.agg_scores()
		self.create_gauge_chart(50, 100, 33, 67, 'Squat.png')
		self.create_gauge_chart(50, 100, 33, 67, 'Core.png')
		self.create_gauge_chart_sym(50, 100, 20, 40, 'KneeStability.png')
		self.create_gauge_chart_sym(self.final_scores[
			"asymmetric_score"], 100, 20, 40, 'Asymmetry.png')


	###################################################################
	######################## Create Output Squat Profile ##############

	# TODO: Delete if not using HTML
	def generate_html(self, data):
		with open('template.html', 'r', encoding='utf-8') as file:
			html_template = file.read()

		# Replace placeholders with data using string formatting
		html_content = html_template.format(title=data['title'], name=data['name'], age=data['age'])
		return html_content

	# TODO: Delete if not using HTML
	def create_pdf_from_data(self):
		# Create original HTML
		data = {
			'title': 'My PDF',
			'name': 'John Doe',
			'age': 30
		}
		html_content = self.generate_html(data)

		# Create output pdf of HTML
		options = {
			'page-size': 'A4',
			'margin-top': '0mm',
			'margin-right': '0mm',
			'margin-bottom': '0mm',
			'margin-left': '0mm',
			'encoding': 'UTF-8',
			'no-outline': None
		}
		pdfkit.from_string(html_content, 'output.pdf', options=options)

	def create_gauge_chart(self, current_value, max_value, red_value,
	                       green_value, filename):
		# Create figure and axis
		fig, ax = plt.subplots(figsize=(8, 2))

		# Define colors
		red_color = '#FE4444'
		yellow_color = '#4444FE'
		green_color = '#44FE44'

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
		yellow_color = '#4444FE'
		green_color = '#44FE44'

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
		background_image = "purple_bg.jpeg"

		# Create a new PDF instance
		pdf = FPDF()
		pdf.add_page()

		pdf.image(background_image, 0, 0, pdf.w,
		          pdf.h)

		pdf.set_text_color(255, 255, 255)
		pdf.set_fill_color(48, 49, 57)
		pdf.rect(0, 0, pdf.w, pdf.h, 'F')  # 'F' means fill the rectangle

		pdf.set_fill_color(23, 24, 32)

		self.draw_rounded_rect(pdf, 8, 36, 194, 84, 10)
		self.draw_rounded_rect(pdf, 8, 132, 94, 154, 10)
		self.draw_rounded_rect(pdf, 108, 132, 94, 64, 10)
		self.draw_rounded_rect(pdf, 108, 200, 94, 86, 10)



		# RGB color
		pdf.set_fill_color(48, 49, 57)
		# Rounded rectangle with 10mm radius for corners
		self.draw_rounded_rect(pdf, 12, 40, 90, 36, 10)
		self.draw_rounded_rect(pdf, 108, 40, 90, 36, 10)
		self.draw_rounded_rect(pdf, 12, 80, 90, 36, 10)
		self.draw_rounded_rect(pdf, 108, 80, 90, 36, 10)

		self.draw_rounded_rect(pdf, 12, 136, 86, 146, 10)
		self.draw_rounded_rect(pdf, 112, 136, 86, 56, 10)



		# Set font and size for the title
		pdf.set_font("Arial", "B", 16)
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


		pdf.image(temp_filename,  116, 208, 39)
		pdf.image(temp_filename_front,  156, 208, 39)

		pdf.image('Squat.png', 7, 60, 96, 10)
		pdf.image('Core.png', 104, 60, 96, 10)
		pdf.image('KneeStability.png', 7, 100, 96, 10)
		pdf.image('Asymmetry.png', 104, 100, 96, 10)

		# Add text content
		pdf.set_font("Arial", "B", 14)
		pdf.cell(0, 20, "Performance Evaluation", ln=True)

		#Add Sub_labels
		pdf.set_font("Arial", "B", 12)
		pdf.set_xy(44, 38)
		pdf.cell(0, 20, "Squat Score", ln=True)
		pdf.set_xy(134, 78)
		pdf.cell(0, 20, "Asymmetry Score", ln=True)
		pdf.set_xy(42, 78)
		pdf.cell(0, 20, "Knee Stability", ln=True)
		pdf.set_xy(138, 38)
		pdf.cell(0, 20, "Core Strength: 100", ln=True)



		# Add text content
		pdf.set_xy(10, 122)
		pdf.set_font("Arial", "B", 14)
		pdf.cell(0, 10, "Recommended Interventions", ln=True)

		pdf.add_page()

		# Save the PDF file
		pdf.output(pdf_path)