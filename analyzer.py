# Created to take in a single subjects squat profile and indicate whicch of
# the potential ailments they exhibit then querying recommendations
# Created by Ben Randoing during 07/2023

import numpy as np
import matplotlib.pyplot as plt
from fpdf import FPDF
from PIL import Image


class AnalyzeSquat():
	ailments = {"not_deep_l": ("Did not reach a squat depth below horizontal L"),
	            "not_deep_r": ("Did not reach a squat depth below horizontal R"),
	            "depth_aym": ("Demonstrated asymmetry at squat base"),
	            "forward_lean": ("Forward lean of trunk"),
	            "backward_lean": ("Backward lean of trunk"),
	            "Varus": ("Demonstrated excess knee varus"),
	            "Valgus": ("Demonstrated excess knee valgus")}

	def __init__(self, hip, knee, ankle, dev, deep, vv, bottom_frame):
		self.hip = hip
		self.knee = knee
		self.ankle = ankle
		self.dev = dev
		self.deep = deep
		self.vv = vv
		self.squat_profile = []
		self.bottom = bottom_frame

	def check_deep(self):
		left = self.deep[0]
		right = self.deep[1]

		if left > -5:
			self.squat_profile.append("not_deep_l")
		if right > -5:
			self.squat_profile.append("not_deep_r")
		if np.abs(left - right) > 5:
			self.squat_profile.append("depth_aym")

	def test(self):
		self.check_deep()

	def make_profile(self):
		pdf_path = "profile.pdf"

		# Create a new PDF instance
		pdf = FPDF()

		# Add a new page
		pdf.add_page()

		# Set font and size for the title
		pdf.set_font("Arial", "B", 16)
		pdf.cell(0, 10, "Functional Movement Assessment", 0, 1, "C")

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
		pdf.cell(0, 20, "Performance Evaluation", ln=True)

		# TODO: Make bullet point list of the Identified ailments

		# Add text content
		pdf.set_xy(10, 120)
		pdf.cell(0, 10, "Performance Evaluation", ln=True)

		# TODO: Make sections with Intervention details for each
		#  identified ailments

		pdf.add_page()

		# Save the PDF file
		pdf.output(pdf_path)



