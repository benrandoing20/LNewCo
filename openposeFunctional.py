# File Created to test OpenPose for Functional Assessment Automation

import cv2

s# Open the video capture device (0 represents the default camera)
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the video capture
    ret, frame = cap.read()

    # If the frame was not captured successfully, exit the loop
    if not ret:
        break

    # Display the frame in a window called "Live Video"
    cv2.imshow('Live Video', frame)

    # Wait for the 'q' key to be pressed to exit the loop
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video capture device and close the window
cap.release()
cv2.destroyAllWindows()
