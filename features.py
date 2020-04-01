from math import log10, copysign
import imutils
from skimage.feature import hog
from skimage import data, exposure
import numpy as np
import cv2 as cv

def CNN(image):
    width = 28
    height = 28
    dim = (width, height)
    #gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # resize image
    resized_image = cv.resize(image, dim, interpolation=cv.INTER_AREA)

    cnn = []
    row, col = resized_image.shape
    for r in range(row):
        for c in range(col):
            cnn.append(int(resized_image[r][c]))

    return cnn

def HOG(image):
    fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                        cells_per_block=(1, 1), visualize=True, multichannel=True)

    # Rescale histogram for better display
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

    width = 64
    height = 64
    dim = (width, height)
    # resize image
    image = cv.resize(hog_image_rescaled, dim, interpolation = cv.INTER_AREA)

    HOG = []
    row, col = image.shape
    for r in range(row):
        for c in range(col):
            HOG.append(float(image[r][c]))

    return HOG

def convex_hull(path):
    global solidity,area
    image = path
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    cnts = cv.findContours(gray.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # loop over the contours
    for (i, c) in enumerate(cnts):
        # compute the area of the contour along with the bounding box
        # to compute the aspect ratio
        area = cv.contourArea(c)
        (x, y, w, h) = cv.boundingRect(c)

        # compute the convex hull of the contour, then use the area of the
        # original contour and the area of the convex hull to compute the
        # solidity
        hull = cv.convexHull(c)
        hullArea = cv.contourArea(hull)
        if hullArea != 0:
            solidity = area / float(hullArea)
        else:
            solidity = -1
        # Draw it
        cv.drawContours(image, [c], -1, (0, 255, 0), 3)

    return area, solidity, image

def Convert_B_W(image):
    #convert black to white and white to black
    n_white_pix = np.sum(image == 255)
    n_black_pix = np.sum(image == 0)
    if n_white_pix > n_black_pix:
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if image[i][j] == 255:
                    image[i][j] = 0
                elif image[i][j] == 0:
                    image[i][j] = 255
        return image
    else:
        return image

def cal_Area(image):
    n_white_pix = np.sum(image == 255)
    n_black_pix = np.sum(image == 0)
    if n_black_pix != 0:
        area = n_white_pix/n_black_pix
        return area
    return -1

def remove_noise(image): 	#Best for Salt & Pepper noise
    r_noise = cv.medianBlur(image, 3)
    return r_noise

def Binarize_image(image):
    # Otsu's thresholding
    ret2, threshold = cv.threshold(image, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    return threshold

def Aspect_Ratio(image):
    x, y, w, h = cv.boundingRect(image)
    if h != 0:
        aspect_ratio = float(w) / h
    else:
        aspect_ratio = -1
    return aspect_ratio

def Moments(image):
    # Calculate Moments
    moments = cv.moments(image) #Binary image

    # Calculate Hu Moments
    huMoments = cv.HuMoments(moments)

    # Log scale hu moments
    moments = []
    for i in range(0, 7):
        try:
            huMoments[i] = -1 * copysign(1.0, huMoments[i]) * log10(abs(huMoments[i]))
            moments.append(huMoments[i])
        except:
            huMoments[i] = 0
            moments.append(huMoments[i])

    return moments