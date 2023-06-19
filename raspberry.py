import cv2
import numpy as np

# Load the template image of a Raspberry Pi with all USB ports
template = cv2.imread('raspberry_pi_template.jpeg', cv2.IMREAD_GRAYSCALE)

# Initialize the camera
camera = cv2.VideoCapture(0)

while True:
    # Display the camera feed
    ret, frame = camera.read()
    cv2.imshow('Camera Feed', frame)

    # Check if the user pressed the Enter key
    if cv2.waitKey(1) == 13:
        break

# Convert the captured frame to grayscale
input_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Apply ORB feature detection and extraction
orb = cv2.ORB_create()
kp_template, des_template = orb.detectAndCompute(template, None)
kp_input, des_input = orb.detectAndCompute(input_image, None)

# Create a brute-force matcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match the descriptors
matches = bf.match(des_template, des_input)
matches = sorted(matches, key=lambda x: x.distance)

# Set a minimum number of good matches
min_good_matches = 20

# If the number of good matches is above the threshold, classify it as a Raspberry Pi with all USB ports
if len(matches) > min_good_matches:
    print("Raspberry Pi with all USB ports")
else:
    print("Raspberry Pi with missing USB ports")

# Release the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
