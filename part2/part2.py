import os
import cv2
import numpy as np


def imgResize(img, width=600):
    # Get image dimensions
    h, w = img.shape[:2]
    # Calculate new height while maintaining aspect ratio
    aspect_ratio = width / float(w)
    new_height = int(h * aspect_ratio)
    # Resize and return the image
    return cv2.resize(img, (width, new_height), interpolation=cv2.INTER_AREA)


def crop(img):
    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply binary threshold
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    # Find external contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Crop image based on the bounding rectangle of the first contou
    if contours:
        x, y, w, h = cv2.boundingRect(contours[0])
        return img[y:y + h - 1, x:x + w - 1]
    
    return img


def detectKP(img):
    descriptor = cv2.SIFT_create()
    # Detect keypoints and compute descriptors
    kps, features = descriptor.detectAndCompute(img, None)
    # Convert keypoints to float32 array
    kps = np.float32([p.pt for p in kps])
    return kps, features


def matchKP(kpA, kpB, fA, fB, ratio):
    # Create a brute-force matcher
    matcher = cv2.BFMatcher()

    # Find initial matches using KNN with k=2
    initMatches = matcher.knnMatch(fA, fB, 2)
    matches = []

    # Apply ratio test to filter good matches
    for match in initMatches:
        if len(match) == 2 and match[0].distance < match[1].distance * ratio:
            matches.append((match[0].trainIdx, match[0].queryIdx))
            
    # Ensure there are enough matches to compute homograph
    if len(matches) > 4:
        # Extract matched keypoints
        pA = np.float32([kpA[i] for (_, i) in matches])
        pB = np.float32([kpB[i] for (i, _) in matches])  

        # Compute homography matrix using RANSAC
        H, status = cv2.findHomography(pA, pB, cv2.LMEDS)
        return matches, H, status
    
    return None


def drawMatches(imgA, imgB, kpA, kpB, matches, status):
    hA, wA = imgA.shape[:2]
    hB, wB = imgB.shape[:2]

    # Create a blank canvas large enough to fit both images side by side
    complete = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")

    # Place both images onto the canvas
    complete[0:hA, 0:wA] = imgA
    complete[0:hB, wA:] = imgB

    # Draw lines between matched keypoints
    for ((idB, idA), s) in zip(matches, status):
        if s == 1:
            # Draw red line between matched points
            ptA = (int(kpA[idA][0]), int(kpA[idA][1]))
            ptB = (int(kpB[idB][0]) + wA, int(kpB[idB][1]))
            cv2.line(complete, ptA, ptB, (0, 0, 255), 1)

    return complete


def stitch(imgA, imgB, ratio=0.6):
    # Detect keypoints and extract features
    kpA, fA = detectKP(imgA)
    kpB, fB = detectKP(imgB)

    # Match keypoints between the two images
    M = matchKP(kpA, kpB, fA, fB, ratio)

    if M is None:
        print("Insufficient matches found")
        return None
    
    matches, H, status = M
    # Warp the first image using the homography matrix to align with the second image
    panorama = cv2.warpPerspective(imgA, H, (imgA.shape[1] + imgB.shape[1], max(imgA.shape[0], imgB.shape[0])))

    # Overlay the second image onto the warped first image
    panorama[0:imgB.shape[0], 0:imgB.shape[1]] = imgB

    # Crop the final stitched image to remove unnecessary black areas
    panorama = crop(panorama)

    # Draw matches between the two images for visualization
    complete = drawMatches(imgA, imgB, kpA, kpB, matches, status)
    return panorama,complete


#clearing output folder
output_folder="out"
for file_name in os.listdir(output_folder):
    file_path = os.path.join(output_folder, file_name)
    
    if os.path.isfile(file_path):
        os.remove(file_path)


input_folder="in2"
NumberOfInputs=4

files=[]
for i in range(NumberOfInputs):
    files.append(f"row-1-column-{NumberOfInputs-i}.jpg")

print(files)
path = [os.path.join(input_folder, f) for f in files]
#loading the 1st image
im1 = cv2.imread(path[0])
im1 = imgResize(im1)

#iteratively stitching images
for i in range(1, len(path)):
    im2 = cv2.imread(path[i])
    im2 = imgResize(im2)
    res = stitch(im1, im2)
    if res is None:
        print("None")
        continue
    
    im1, complete = res    
    cv2.imwrite(os.path.join(output_folder, f"match{i}.jpg"), complete)

cv2.imwrite(os.path.join(output_folder, f"panorama.jpg"), im1)


