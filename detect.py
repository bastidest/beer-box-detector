import cv2
import numpy as np

def detect(path):
    img = cv2.imread(path)
    # resize = cv2.resize(img, None, fx=0.3, fy=0.3, interpolation = cv2.INTER_CUBIC)
    resize = cv2.resize(img, (400, 300), interpolation = cv2.INTER_CUBIC)
    gray = cv2.cvtColor(resize, cv2.COLOR_BGR2GRAY)
    _,thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    # contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # cv2.drawContours(resize, contours, -1, (0, 255, 0), 3)

    # Copy edges to the images that will display the results in BGR
    dst = cv2.Canny(thresh, 100, 180, None, 3)

    cdstP = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)

    linesP = cv2.HoughLinesP(dst, rho = 1, theta = 1*np.pi/180, threshold = 100, minLineLength = 150, maxLineGap = 200)

    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv2.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)


    # cv2.imshow('Calculated Output', thresh)
    # cv2.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)
    threshrgb = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    imstack = np.hstack((threshrgb, cdstP))
    cv2.imshow(path, imstack)


detect('./samples/kasten1.jpg')
detect('./samples/kasten2.jpg')
detect('./samples/kasten3.jpg')
detect('./samples/kasten4.jpg')
detect('./samples/kasten5.jpg')
detect('./samples/kasten6.jpg')
cv2.waitKey()
