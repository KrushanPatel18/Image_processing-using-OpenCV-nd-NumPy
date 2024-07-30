import cv2
import numpy as np

def convert_to_grayscale(image):
    """Convert an image to grayscale."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def apply_filter(image, filter_type):
    """Apply a specific filter to the image."""
    if filter_type == 'blur':
        return cv2.GaussianBlur(image, (5, 5), 0)
    elif filter_type == 'sharpen':
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        return cv2.filter2D(image, -1, kernel)
    else:
        return image

def detect_edges(image):
    """Detect edges using the Canny edge detector."""
    return cv2.Canny(image, 100, 200)

def find_contours_and_draw(original_image, edges):
    """Find contours in the image and draw them."""
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour_image = original_image.copy()
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
    return contour_image
