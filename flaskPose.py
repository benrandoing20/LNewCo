# Created by Ben Randoing on 7/5/2023 to host a mediapipe
# pose estimator for angle visualization


from flask import Flask, jsonify, request, render_template, redirect, \
    url_for, make_response, Response
from helperdb import angle_diff, frame
# from flask_login import UserMixin, LoginManager, login_required, login_user, current_user
# from flask_wtf import FlaskForm
# from wtforms import SubmitField, SelectField
# from werkzeug.security import generate_password_hash, check_password_hash
# from wtforms.validators import DataRequired

import numpy as np
import base64
import json
import cv2
import mediapipe as mp
from PIL import Image
import io

app = Flask(__name__)

# Initialize MediaPipe Drawing
mp_drawing = mp.solutions.drawing_utils

# Initialize MediaPipe Pose model
mp_pose = mp.solutions.pose

############################################################################
########################## Flask Endpoints #################################
############################################################################

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/capture', methods=['POST'] )
def capture():
    if request.method == 'POST':
        image_data = request.form['image'].split(",")[1]
        decoded_data = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(decoded_data))
        image_np = np.array(image)

        frame, angle_diff = generate_frame(image_np)
        return jsonify({'angle_diff': angle_diff})

############################################################################
########################## Helper Functions ################################
############################################################################

def generate_frame(frame):
    ## Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.5,
                      min_tracking_confidence=0.5) as pose:

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

                angle_diff = angle_trunk - angle_shank

                # Visualize angle of trunk and shank on video wrt ground
                cv2.putText(image, str(angle_trunk),
                            tuple(np.multiply(hip, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2,
                            cv2.LINE_AA)

                cv2.putText(image, str(angle_shank),
                            tuple(np.multiply(knee, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2,
                            cv2.LINE_AA)

                print(angle_trunk)


            except:
                pass

            # Render detections
            # mp_drawing.draw_landmarks(image, results.pose_landmarks,
            #                           mp_pose.POSE_CONNECTIONS,
            #                           mp_drawing.DrawingSpec(color=(245, 117, 66),
            #                                                  thickness=2,
            #                                                  circle_radius=2),
            #                           mp_drawing.DrawingSpec(color=(245, 66, 230),
            #                                                  thickness=2,
            #                                                  circle_radius=2)
            #                       )


            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()
            return (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'), angle_diff

            # if cv2.waitKey(10) & 0xFF == ord('q'):
            #     break

            # cap.release()
            # cv2.destroyAllWindows()


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
    app.run(host="0.0.0.0", port=5050)
