import cv2
import numpy as np

# Import necessary libraries

# Function to detect and label lines in an image
def detect_and_label_lines(image_path):
    # Load the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Error: Unable to load image.")
        return

    # Apply edge detection using Canny
    edges = cv2.Canny(image, 50, 150, apertureSize=3)

    # Detect lines using Hough Line Transform
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)

    # Convert the grayscale image to BGR for visualization
    output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Check if lines are detected
    if lines is not None:
        for i, line in enumerate(lines):
            rho, theta = line[0]
            # Convert polar coordinates to Cartesian coordinates
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            # Draw the line on the image
            cv2.line(output_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # Label the line with its index
            label_position = (int(x0), int(y0))
            cv2.putText(output_image, f"Line {i+1}", label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # Display the result
    cv2.imshow("Detected Lines", output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Main function to execute the program
if __name__ == "__main__":
    # Path to the input image
    image_path = "input_image.jpg"  # Replace with your image path

    # Call the function to detect and label lines
    detect_and_label_lines(image_path)