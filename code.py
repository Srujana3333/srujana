import cv2
import os
import numpy as np
# Input and output folders
input_folder = "input"
output_folder = "output"

# Ensure the output folder exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Function to calculate total surface area of a contour
def calculate_surface_area(contour):
    return cv2.contourArea(contour)

# Function to process an image
def process_image(filename):
    # Read the image
    img = cv2.imread(filename)

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to segment particles (adjust threshold value as needed)
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

    # Find contours in the binary image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw the smallest encapsulating circle, major axis, total surface area, total perimeter, and centroids for each particle
    img_processed = img.copy()
    for contour in contours:
        # Calculate the minimum enclosing circle
        (x, y), radius = cv2.minEnclosingCircle(contour)
        center = (int(x), int(y))
        radius = int(radius)

        # Calculate the surface area
        area = calculate_surface_area(contour)

        ellipse = None
        # Check if there are at least 5 points to fit an ellipse
        if len(contour) >= 5:
        # Fit an ellipse to the contour
            ellipse = cv2.fitEllipse(contour)
        if ellipse is not None:
        # Calculate the major axis
            major_axis = max(ellipse[1])

        else:
            # If there are fewer than 5 points to fit an ellipse, estimate major axis as the longest distance between any two points
            points = np.squeeze(contour)  # Remove extra dimensions
            distances = [np.linalg.norm(p1 - p2) for p1 in points for p2 in points]
            major_axis = max(distances)

        # Draw the major axis
        #if major_axis > 0:
        cv2.ellipse(img_processed, ellipse, (0, 0, 255), 2)  # Red ellipse

        # Display the major axis length as text
        M = cv2.moments(contour)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        cv2.putText(img_processed, f"Major Axis Length: {major_axis:.2f}", (cx - 60, cy + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255), 2)
        #else:
        #    major_axis = 0

        # Calculate the perimeter of the contour
        perimeter = cv2.arcLength(contour, closed=True)

        # Calculate the centroid of the contour
        M = cv2.moments(contour)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        centroid = (cx, cy)

        # Draw the minimum enclosing circle
        cv2.circle(img_processed, center, radius, (0, 0, 255), 2)  # Red circle



        # Display the total surface area as text
        cv2.putText(img_processed, f"Total surface area: {area:.2f}", (cx - 40, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Display the total perimeter as text
        cv2.putText(img_processed, f"Total Perimeter: {perimeter:.2f}", (cx - 60, cy + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Draw a marker at the centroid
        cv2.circle(img_processed, centroid, 5, (0, 0, 255), -1)  # Red circle
        # Display the centroid as text
        cv2.putText(img_processed, f"Centroid: ({cx},{cy})", (cx - 40, cy + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255), 2)

    # Save the processed image with analysis
    output_filename = os.path.join(output_folder, os.path.basename(filename))
    cv2.imwrite(output_filename, img_processed)

# Process each image in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith((".jpg", ".png")):
        input_image_path = os.path.join(input_folder, filename)
        process_image(input_image_path)

print("Processing complete.")
