import cv2
import numpy as np

IMG_WIDTH = 400
IMG_HEIGHT = 300

def get_small_img(path):
    img = cv2.imread(path)
    return cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT), interpolation = cv2.INTER_CUBIC)
    
def detect_straight_lines(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _,thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    blurred = cv2.blur(thresh, (11, 11))

    traced = cv2.Canny(blurred, 50, 200, None, 3)

    # cdstP = cv2.cvtColor(traced, cv2.COLOR_GRAY2BGR)

    lines = cv2.HoughLinesP(traced, rho = 1, theta = 1*np.pi/180, threshold = 70, minLineLength = 150, maxLineGap = 200)

    if lines is not None:
        for i in range(0, len(lines)):
            l = lines[i][0]
            cv2.line(img, (l[0], l[1]), (l[2], l[3]), (0,0,255), 1, cv2.LINE_AA)

    tracedrgb = cv2.cvtColor(traced, cv2.COLOR_GRAY2BGR)
    imstack = np.hstack((img, tracedrgb))
    cv2.imshow(path, imstack)
    return lines

def detect_box(name, lines):
    alignments = align_lines(lines)
    print_aligned_lines(name, alignments)
    bounds = get_outer_bounds(alignments)
    print_bounds(name, bounds)

#   +-------+-------+            I 
#   |       |       |        ----------
#   |   I   |  II   |      |            |
#   |       |       |      |            |
#   +-------+-------+   IV |            | II
#   |       |       |      |            |   
#   |  IV   |  III  |      |            |   
#   |       |       |       -----------     
#   +-------+-------+           III         
# 
#       quadrants            alignments
#
def align_lines(lines):
    ret = ([], [], [], [])
    for line in lines:
        line = line[0]
        quadrants = (get_quadrant_of_point((line[0], line[1])), get_quadrant_of_point((line[2], line[3])))
        alignment = get_line_alignment(quadrants)
        if alignment == 0:
            continue
        ret[alignment - 1].append(line)
    return ret

def get_quadrant_of_point(point):
    if point[0] < IMG_WIDTH / 2:
        if point[1] < IMG_HEIGHT / 2:
            return 1
        else:
            return 4
    else:
        if point[1] < IMG_HEIGHT / 2:
            return 2
        else:
            return 3

def get_line_alignment(quadrants):
    quadset = set(quadrants)
    if quadset == {1,2}:
        return 1
    if quadset == {2,3}:
        return 2
    if quadset == {3,4}:
        return 3
    if quadset == {4,1}:
        return 4
    return 0

def print_aligned_lines(name, alignments):
    empty = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), np.uint8)
    for line in alignments[0]:
        cv2.line(empty, (line[0], line[1]), (line[2], line[3]), (0, 0, 255), 1, cv2.LINE_AA)
    for line in alignments[1]:
        cv2.line(empty, (line[0], line[1]), (line[2], line[3]), (255, 0, 255), 1, cv2.LINE_AA)
    for line in alignments[2]:
        cv2.line(empty, (line[0], line[1]), (line[2], line[3]), (0, 255, 255), 1, cv2.LINE_AA)
    for line in alignments[3]:
        cv2.line(empty, (line[0], line[1]), (line[2], line[3]), (255, 255, 0), 1, cv2.LINE_AA)
    cv2.imshow(name, empty)

def get_outer_bounds(alignments):
    top = min(alignments[0], key=lambda e: (e[1] + e[3]) / 2, default=(0,0,0,0))
    right = max(alignments[1], key=lambda e: (e[0] + e[2]) / 2, default=(0,0,0,0))
    bottom = max(alignments[2], key=lambda e: (e[1] + e[3]) / 2, default=(0,0,0,0))
    left = min(alignments[3], key=lambda e: (e[0] + e[2]) / 2, default=(0,0,0,0))
    ret = (top, right, bottom, left)
    return ret

def print_bounds(name, bounds):
    print_aligned_lines(name, (
        [bounds[0]],
        [bounds[1]],
        [bounds[2]],
        [bounds[3]],
    ))


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
    lines = detect_straight_lines(resized)
    shape = detect_box(path, lines)

cv2.waitKey()
