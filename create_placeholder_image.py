import cv2
import numpy as np

# Define image dimensions and color
width, height = 640, 480
black_color = (0, 0, 0)

# Create a black image
image = np.zeros((height, width, 3), dtype=np.uint8)
image[:] = black_color

# Save the image
cv2.imwrite("test_image.jpg", image)

print("Placeholder image 'test_image.jpg' created successfully.")
