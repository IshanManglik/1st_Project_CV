import cv2
import numpy as np

# Load the template image of a Raspberry Pi with all ports
template = cv2.imread('raspberry_pi_template.jpeg', cv2.IMREAD_GRAYSCALE)

# Define the path to the input image of the Raspberry Pi to classify
input_image_path = 'raspberry_pi_to_classify.png'

# Load the input image
input_image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)

# Apply template matching to find the template in the input image
result = cv2.matchTemplate(input_image, template, cv2.TM_CCOEFF_NORMED)
_, max_val, _, max_loc = cv2.minMaxLoc(result)

# Set a threshold for the match score to determine if the template is found
threshold = 0.8

# If the match score is above the threshold, classify it as a Raspberry Pi with all ports
if max_val > threshold:
    print("Raspberry Pi with all ports")

    # Draw a bounding box around the template in the input image
    template_height, template_width = template.shape
    top_left = max_loc
    bottom_right = (top_left[0] + template_width, top_left[1] + template_height)
    cv2.rectangle(input_image, top_left, bottom_right, 255, 2)

else:
    print("Raspberry Pi with missing ports")

# Display the input image with the bounding box (if applicable)
cv2.imshow('Raspberry Pi Classification', input_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
