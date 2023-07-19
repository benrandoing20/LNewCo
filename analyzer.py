# Created to take in a single subjects squat profile and indicate whicch of
# the potential ailments they exhibit then querying recommendations
# Created by Ben Randoing during 07/2023

import numpy as np
import matplotlib.pyplot as plt
from fpdf import FPDF
from PIL import Image


class AnalyzeSquat():
	def __init__(self, hip, knee, ankle, dev, vv, deep, bottom_frame):
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
		self.ailments_store = {
			"not_deep_l": ["Did not reach a squat depth below horizontal L"],
			"not_deep_r": ["Did not reach a squat depth below horizontal R"],
			"depth_aym": ["Demonstrated asymmetry at squat base",
			              "We recommend rolling out the weak side adductor "
			              "and IT-Band as well as the strong side Biceps "
			              "Femoris, Gastrocnemius, Soleus, and Piriformis. "
			              "Please also consider strengthening exercises "
			              "including weak side leg pulls and strong side "
			              "leg pushes."],
			"forward_lean": ["Forward lean of trunk"],
			"backward_lean": ["Backward lean of trunk"],
			"Varus": ["Demonstrated excess knee varus"],
			"Valgus": ["Demonstrated excess knee valgus"]}

	def check_deep(self):
		left = self.deep[0]
		right = self.deep[1]

		if left > -5:
			self.squat_profile.append("not_deep_l")
		if right > -5:
			self.squat_profile.append("not_deep_r")
		if np.abs(left - right) > 5:
			self.squat_profile.append("depth_aym")

	def create_bullet_list(self):
		for i, val in enumerate(self.squat_profile):
			self.ailments.append(self.ailments_store[val][0])

	def create_recs_dictionary(self):
		for i, val in enumerate(self.squat_profile):
			self.interventions[self.ailments_store[val][0]] = (
				self.ailments_store[val][1])


	def test(self):
		self.check_deep()
		self.create_bullet_list()
		self.create_recs_dictionary()

	def bullet_point_list(self, items, pdf):
		pdf.set_font("Arial", "B", 10)
		for item in items:
			pdf.cell(0, 10, "- " + item, ln=True)

	def headed_bullet_point_list(self, items_dict, pdf):
		pdf.set_font("Arial", "B", 10)
		for key, value in items_dict:
			pdf.cell(0, 10, "- " + key, ln=True)
			pdf.cell(0, 10, "- " + value, ln=True)

	def make_profile(self):
		pdf_path = "profile.pdf"

		# Create a new PDF instance
		pdf = FPDF()
		pdf.add_page()

		# Set font and size for the title
		pdf.set_font("Arial", "B", 16)
		pdf.cell(0, 10, "Functional Movement Profile", 0, 1, "C")

		pdf.set_font("Arial", "", 12)

		# Create a PIL Image object from the CV2 frame
		image_pil = Image.fromarray(self.bottom_frame)
		temp_filename = 'temp_image.jpg' # Save the PIL image as a temporary file
		image_pil.save(temp_filename)
		pdf.image(temp_filename, 147, 45, 43, 80)

		# Add text content
		pdf.set_font("Arial", "B", 14)
		pdf.cell(0, 20, "Performance Evaluation", ln=True)

		# Add content (including bullet point list)
		self.bullet_point_list(self.ailments, pdf)

		# Add text content
		pdf.set_xy(10, 120)
		pdf.set_font("Arial", "B", 14)
		pdf.cell(0, 10, "Recommended Interventions", ln=True)

		# Add content (including bullet point list)
		self.headed_bullet_point_list(self.interventions, pdf)

		pdf.add_page()

		# Save the PDF file
		pdf.output(pdf_path)





