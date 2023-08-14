# Created toStore Functions for Calculation of Functional Movement Angles
# Created by Ben Randoing for L-NewCo during 07/2023

import numpy as np

def get_hip_angle(landmarks, mp_pose):
	'''
	Hip Kinematics
	'''
	# Get coordinates Left
	knee_l = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
	          landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]

	hip_l = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
	         landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]

	shoulder_l = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
	              landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]

	# Get coordinates Right
	knee_r = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
	          landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]

	hip_r = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
	         landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]

	# Get coordinates Upper Body
	shoulder_r = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
	              landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]

	# Calculate angles
	hip_angle_left = calculate_angle(knee_l, hip_l, shoulder_l)
	hip_angle_right = calculate_angle(knee_r, hip_r, shoulder_r)

	return hip_angle_left, hip_angle_right

def get_knee_angle(landmarks, mp_pose):
	'''
	Knee Kinematics
	'''
	# Get coordinates Left
	knee_l = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
	          landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]

	hip_l = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
	         landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]

	ankle_l = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
	           landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

	# Get coordinates Right
	knee_r = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
	          landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]

	hip_r = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
	         landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]

	ankle_r = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
	           landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

	# Calculate angles
	knee_angle_left = calculate_angle(hip_l, knee_l, ankle_l)
	knee_angle_right = calculate_angle(hip_r, knee_r, ankle_r)

	return knee_angle_left, knee_angle_right


def get_ankle_angle(landmarks, mp_pose):
	'''
	Ankle Kinematics
	'''
	# Get coordinates Left
	knee_l = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
	          landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]

	foot_l = [landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x,
	         landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y]

	ankle_l = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
	           landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

	# Get coordinates Right
	knee_r = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
	          landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]

	foot_r = [landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x,
	         landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y]

	ankle_r = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
	           landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

	# Calculate angles
	ankle_angle_left = calculate_angle(knee_l, ankle_l, foot_l)
	ankle_angle_right = calculate_angle(knee_r, ankle_r, foot_r)

	return ankle_angle_left, ankle_angle_right


def get_hipshin_angle(landmarks, mp_pose):
	'''
	Deviation Angle between Trunk Plane and Upper Leg PLane
	'''
	torso_landmarks = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
	                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
	                          landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
	                          landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]]

	shin_landmarks = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
	                         landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
	                         landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value],
	                         landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]]

	# Extract the x, y, and z coordinates of the landmarks
	torso_coordinates = [(value.x, value.y, value.z) for value in
	                     torso_landmarks]
	shin_coordinates = [(value.x, value.y, value.z) for value in
	                    shin_landmarks]

	# Calculate the vector of the torso plane
	torso_vector1 = np.subtract(torso_coordinates[0], torso_coordinates[1])
	torso_vector2 = np.subtract(torso_coordinates[2], torso_coordinates[1])

	# Calculate the vector of the shin plane
	shin_vector1 = np.subtract(shin_coordinates[0], shin_coordinates[1])
	shin_vector2 = np.subtract(shin_coordinates[2], shin_coordinates[1])

	# Calculate the angle between the two planes
	torso_plane_normal = np.cross(torso_vector1, torso_vector2)
	shin_plane_normal = np.cross(shin_vector1, shin_vector2)

	deviation_angle = calculate_angle_plane(torso_plane_normal,
	                                      shin_plane_normal)
	return deviation_angle


def get_deepfemur_angle(landmarks, mp_pose):
	'''
	Angle of Upper Leg Segment at Deepest Part of Squat
	'''
	# Get coordinates Left
	knee_l = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
	          landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]

	hip_l = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
	          landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]

	# Get coordinates Right
	knee_r = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
	          landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]

	hip_r = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
	          landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]

	# Calculate angles
	femur_angle_left = -1 * calculate_angle_horz(hip_l, knee_l)
	femur_angle_right = -1 * calculate_angle_horz(hip_r, knee_r)

	return femur_angle_left, femur_angle_right


def get_varvalg_angle(landmarks, mp_pose):
	'''
	Angle of Upper Leg Segment at Deepest Part of Squat
	'''
	# Get coordinates Left
	knee_l = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
	          landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]

	ankle_l = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
	          landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

	# Get coordinates Right
	knee_r = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
	          landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]

	ankle_r = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
	          landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

	# Calculate angles
	varvalg_angle_left = calculate_angle_vert(ankle_l, knee_l)
	varvalg_angle_right = calculate_angle_vert(ankle_r, knee_r)

	return varvalg_angle_left, varvalg_angle_right

def get_shoulder_angle(landmarks, mp_pose):
	'''
	Deviation Angle between Trunk Plane and Upper Leg PLane
	'''
	torso_landmarks = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
	                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
	                          landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
	                          landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]]

	arms_landmarks = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
	                         landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
	                         landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
	                         landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]]

	# Extract the x, y, and z coordinates of the landmarks
	torso_coordinates = [(value.x, value.y, value.z) for value in
	                     torso_landmarks]
	arm_coordinates = [(value.x, value.y, value.z) for value in
	                    arms_landmarks]

	# Calculate the vector of the torso plane
	torso_vector1 = np.subtract(torso_coordinates[0], torso_coordinates[1])
	torso_vector2 = np.subtract(torso_coordinates[2], torso_coordinates[1])

	# Calculate the vector of the shin plane
	arm_vector1 = np.subtract(arm_coordinates[0], arm_coordinates[1])
	arm_vector2 = np.subtract(arm_coordinates[2], arm_coordinates[1])

	# Calculate the angle between the two planes
	torso_plane_normal = np.cross(torso_vector1, torso_vector2)
	arm_plane_normal = np.cross(arm_vector1, arm_vector2)

	shoulder_deviation_angle = calculate_angle_plane(torso_plane_normal,
	                                      arm_plane_normal)

	shoulder_deviation_angle = 180 - shoulder_deviation_angle # must adjust
	# due to angle right vector being positive

	return shoulder_deviation_angle

def get_foot_angle(landmarks, mp_pose):
	'''
	Angle of Foot (Toe - Heel) with respect to horizontal
	'''
	# Get coordinates Left
	heel_l = [landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].x,
	          landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].y]

	toe_l = [landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x,
	          landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y]

	# Get coordinates Right
	heel_r = [landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].x,
	          landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].y]

	toe_r = [landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x,
	          landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y]

	# Calculate angles
	foot_angle_left = calculate_angle_horz(heel_l, toe_l)
	foot_angle_right = calculate_angle_horz(heel_r, toe_r)

	return foot_angle_left, foot_angle_right

def get_inout_angle(landmarks, mp_pose):
	'''
	Angle of Foot (Toe - Heel) with respect to horizontal
	'''
	# Get coordinates Left
	heel_l = [landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].x,
	          landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].z]

	toe_l = [landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x,
	          landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].z]

	# Get coordinates Right
	heel_r = [landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].x,
	          landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].z]

	toe_r = [landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x,
	          landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].z]

	# Calculate angles
	foot_angle_left = calculate_angle_in(heel_l, toe_l)
	foot_angle_right = calculate_angle_in(heel_r, toe_r)

	return foot_angle_left, foot_angle_right



####################################################################
################## Angle Calculating Functions #####################
####################################################################

def calculate_angle(point1, point2, point3):
	'''
	Calculate angle between two lines given three points
	'''
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
    '''
    Calculate angle between lines and ground
    '''
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

        # Determine the sign of the angle
        orientation = np.sign(vector1[1])
        degree_angle *= orientation

        if degree_angle > 180.0:
            degree_angle = 360 - degree_angle # keeps between 0 and 180
        # elif degree_angle < 0:
        #     degree_angle = (degree_angle + 180) * -1

        return degree_angle

    except ZeroDivisionError:
        return 90.0

def calculate_angle_vert(point1, point2):
	'''
	Calculate angle between lines and ground
	'''
	if point1 == (0, 0) or point2 == (0, 0):
		return 0

	# Calculate vector
	vector1 = (point1[0] - point2[0], point1[1] - point2[1])
	vector2 = (0, 1)

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

		# Determine the sign of the angle
		orientation = np.sign(vector1[0])
		degree_angle *= orientation

		if degree_angle > 180.0:
			degree_angle = 360 - degree_angle

		return degree_angle

	except ZeroDivisionError:
		return 90.0

def calculate_angle_in(point1, point2):
	'''
	Calculate angle between lines and ground
	'''
	if point1 == (0, 0) or point2 == (0, 0):
		return 0

	# Calculate vector
	vector1 = (point1[0] - point2[0], point1[1] - point2[1])
	vector2 = (0, 1)

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

		# Determine the sign of the angle
		orientation = np.sign(vector1[1])
		degree_angle *= orientation

		# if degree_angle > 180.0:
		# 	degree_angle = 360 - degree_angle

		return degree_angle

	except ZeroDivisionError:
		return 90.0

def calculate_angle_plane(plane1, plane2):
    '''
    Calculate angle between lines and ground
    '''
    try:
        dot_product = np.dot(plane1, plane2)
        norm_product = np.linalg.norm(plane1) * np.linalg.norm(plane2)
        angle_radians = np.arccos(dot_product / norm_product)
        angle_degrees = np.degrees(angle_radians)

        # Check if the cross product is negative
        cross_product = np.cross(plane1, plane2)
        if cross_product[2] < 0:
            angle_degrees = -angle_degrees

    except ZeroDivisionError:
        return 90.0

    return angle_degrees