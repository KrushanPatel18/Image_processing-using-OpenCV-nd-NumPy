import cv2
import numpy as np
from utils import (
    convert_to_grayscale,
    apply_filter,
    detect_edges,
    find_contours_and_draw
)

def main():
    # Load an image from file
    image_path = r'C:\Users\DELL\Desktop\image_processing\Lenna_(test_image).png'
    image = cv2.imread(image_path)
    
    if image is None:
        print("Error: Image not found.")
        return
    
    # Convert the image to grayscale
    gray_image = convert_to_grayscale(image)
    
    # Apply Gaussian blur
    blurred_image = apply_filter(gray_image, 'blur')
    
    # Apply sharpening
    sharpened_image = apply_filter(gray_image, 'sharpen')
    
    # Detect edges using Canny edge detector
    edges = detect_edges(gray_image)
    
    # Find contours and draw them on the image
    contour_image = find_contours_and_draw(image, edges)
    
    # Save the processed images
    cv2.imwrite(r'C:\Users\DELL\Desktop\image_processing\gray_image.jpg', gray_image)
    cv2.imwrite(r'C:\Users\DELL\Desktop\image_processing\blurred_image.jpg', blurred_image)
    cv2.imwrite(r'C:\Users\DELL\Desktop\image_processing\sharpened_image.jpg', sharpened_image)
    cv2.imwrite(r'C:\Users\DELL\Desktop\image_processing\edges.jpg', edges)
    cv2.imwrite(r'C:\Users\DELL\Desktop\image_processing\contour_image.jpg', contour_image)
    
    # Display the original and processed images
    cv2.imshow('Original Image', image)
    cv2.imshow('Grayscale Image', gray_image)
    cv2.imshow('Blurred Image', blurred_image)
    cv2.imshow('Sharpened Image', sharpened_image)
    cv2.imshow('Edges', edges)
    cv2.imshow('Contour Image', contour_image)
    
    # Wait for a key press and close all windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
