import cv2
import numpy as np
import os

def clearOutput(path):
    for folder in path:
        for file_name in os.listdir(folder):
            file_path = os.path.join(folder, file_name)
            
            if os.path.isfile(file_path):
                os.remove(file_path)


def resize(img):
    # Determine the maximum dimension (height or width) of the image
    max_size = max(img.shape[:2]) 

    # Scale the image down if its largest dimension exceeds 700 pixels
    scale = 700 / max_size if max_size > 700 else 1
    # Resize the image while maintaining the aspect ratio
    img_resized = cv2.resize(img, (0, 0), fx=scale, fy=scale)

    return img_resized,scale


def binImg(img):# Convert the resized image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
    # Apply Gaussian blur to reduce noise and smooth the image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Apply adaptive thresholding to create a binary image
    binary_img = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    kernel = np.ones((3, 3), np.uint8)
    # Apply morphological closing to fill small gaps in the binary image
    closed = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel, iterations=2)

    return closed


def detectCircles(binary_img,scale):
    detected_objects, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # contours = img.copy()

    min_area = 500 * (scale ** 2)
    circular_shapes = []
    # Iterate through detected contours
    for obj in detected_objects:
        perimeter = cv2.arcLength(obj, True)
        area = cv2.contourArea(obj)
        #remove erroneous contours
        if perimeter > 0:
            circularity = 4 * np.pi * (area / (perimeter ** 2))
            if circularity < 1.2 and area > min_area:
                circular_shapes.append(obj)

    return circular_shapes


def drawContours(img,circular_shapes):
    result_img = img.copy()
        
    # Draw contours around detected circular shapes
    cv2.drawContours(result_img, circular_shapes, -1, (0, 0, 255), 2)
    save_path="outline/outline.jpeg"
    cv2.imwrite(save_path, result_img)

    result_img = img.copy()
    # Create a blank mask
    mask_layer = np.zeros(result_img.shape[:2], dtype=np.uint8)
    # mask_3ch = cv2.cvtColor(mask_layer, cv2.COLOR_GRAY2BGR)
    # masked_result = cv2.bitwise_and(img, mask_3ch)
    
    # Fill detected circular shapes in the mask
    cv2.drawContours(mask_layer, circular_shapes, -1, 255, thickness=cv2.FILLED)
    # Apply the mask to the image to isolate detected shapes
    result_img = cv2.bitwise_and(result_img, result_img, mask=mask_layer)

    save_path="contours/contours.jpeg"
    cv2.imwrite(save_path, result_img)


def segment_count(img,circular_shapes):
    segmented_coins = []
    # Loop through each detected circular shape
    for i, cnt in enumerate(circular_shapes):
        # Find the minimum enclosing circle for the contour
        (x, y), r = cv2.minEnclosingCircle(cnt)
        c = (int(x), int(y)) 
        r = int(r)
        # Create a blank mask
        mask = np.zeros_like(img, dtype=np.uint8)
        # Draw a filled white circle on the mask to isolate the coin
        cv2.circle(mask, c, r, (255, 255, 255), -1) 
        # Apply the mask to extract the coin region from the image
        coin_segment = cv2.bitwise_and(img, mask)
        
        x1, y1 = c[0] - r, c[1] - r
        x2, y2 = c[0] + r, c[1] + r
        # Crop the coin region from the masked image
        coin_segment = coin_segment[y1:y2, x1:x2]
        segmented_coins.append(coin_segment)
        
        coin_path = os.path.join("segmentedCoins", f"coin_{i+1}.jpg")
        cv2.imwrite(coin_path, coin_segment)

    # Print the total number of segmented coins
    print("Total number of coins in the image is: ",i+1)


#clearing output folder
clearOutput(["contours","outline","segmentedCoins"])


img=cv2.imread("coins.jpeg")

if img is None:
    print("Error: Image not found or unable to load")
else:
    img,scale=resize(img)
    binary_img=binImg(img)
    circular_shapes=detectCircles(binary_img,scale)
    drawContours(img,circular_shapes)
    segment_count(img,circular_shapes)