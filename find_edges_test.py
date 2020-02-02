import math
import cv2
import numpy as np

IMG_WIDTH = 1600
IMG_HEIGHT = 1200
NR_BOTTLES_WIDE = 5
NR_BOTTLES_NARROW = 4
PADDING_WIDE = 0.05
PADDING_NARROW = 0.032



def get_small_img(path):
    img = cv2.imread(path)
    return cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT), interpolation = cv2.INTER_CUBIC)
    
def detect_straight_lines(img):
    blurred = cv2.GaussianBlur(img, (0,0), 3.5)
    sobelx = np.absolute(cv2.Sobel(blurred,cv2.CV_32F,1,0,ksize=3)) / 255
    sobely = np.absolute(cv2.Sobel(blurred,cv2.CV_32F,0,1,ksize=3)) / 255
    # sobelx = cv2.cvtColor(sobelx, cv2.COLOR_BGR2GRAY)
    # sobely = cv2.cvtColor(sobely, cv2.COLOR_BGR2GRAY)
    mag_color, ang = cv2.cartToPolar(sobelx, sobely)
    mag = mag_color
    mag = mag * 255
    mag = mag.astype(np.uint8)
    mag = cv2.cvtColor(mag, cv2.COLOR_BGR2GRAY)
    
    # # thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 10)
    # _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    # blurred = cv2.blur(thresh, (11, 11))

    traced = cv2.Canny(mag, PARAM_CANNY_THRESHOLD, PARAM_CANNY_THRESHOLD, None, 3)


    lines = cv2.HoughLines(traced, rho = 1, theta = 1*np.pi/180, threshold = 200)
    # Draw the lines
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            cv2.line(mag_color, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)

    # if lines is not None:
    #     for i in range(0, len(lines)):
    #         l = lines[i][0]
    #         cv2.line(mag_color, (l[0], l[1]), (l[2], l[3]), (0,0,255), 1, cv2.LINE_AA)

    # tracedrgb = cv2.cvtColor(traced, cv2.COLOR_GRAY2BGR)
    # imstack = np.hstack((img, test))
    cv2.imshow(path, mag_color)
    # return lines

paths = [
    './samples/kasten1.jpg',
    './samples/kasten2.jpg',
    './samples/kasten3.jpg',
    './samples/kasten4.jpg',
    './samples/kasten5.jpg',
    './samples/kasten6.jpg',
]

for path in paths:
    print(path)
    resized = get_small_img(path)
    detect_straight_lines(resized)

cv2.waitKey()
