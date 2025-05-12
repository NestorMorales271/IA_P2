import cv2
import numpy as np

# Import necessary libraries
import matplotlib.pyplot as plt

# Load the input image
# Replace 'input_image.jpg' with the path to your image
image_path = 'input_image.jpg'
image = cv2.imread(image_path, cv2.IMREAD_COLOR)

# Check if the image was loaded successfully
if image is None:
    raise FileNotFoundError(f"Image at path '{image_path}' not found.")

# Convert the image to grayscale
# This simplifies edge detection by reducing the image to a single channel
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian Blur to reduce noise
# This step helps in improving edge detection by smoothing the image
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

# Perform edge detection using the Canny algorithm
# The thresholds can be adjusted to detect more or fewer edges
edges = cv2.Canny(blurred_image, threshold1=50, threshold2=150)

# Perform segmentation using thresholding
# This creates a binary image where pixels are either 0 or 255
_, segmented_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

# Display the results
# Use matplotlib to show the original, edges, and segmented images
plt.figure(figsize=(10, 7))

# Show the original image
plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')

# Show the edges detected
plt.subplot(1, 3, 2)
plt.title("Edge Detection")
plt.imshow(edges, cmap='gray')
plt.axis('off')

# Show the segmented image
plt.subplot(1, 3, 3)
plt.title("Segmentation")
plt.imshow(segmented_image, cmap='gray')
plt.axis('off')

# Display all images
plt.tight_layout()
plt.show()