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

    lines = cv2.HoughLinesP(traced, rho = 1, theta = 1*np.pi/180, threshold = 50, minLineLength = 150, maxLineGap = 200)

    # if lines is not None:
    #     for i in range(0, len(lines)):
    #         l = lines[i][0]
    #         cv2.line(img, (l[0], l[1]), (l[2], l[3]), (0,0,255), 1, cv2.LINE_AA)

    # tracedrgb = cv2.cvtColor(traced, cv2.COLOR_GRAY2BGR)
    # imstack = np.hstack((img, tracedrgb))
    # cv2.imshow(path, imstack)
    return lines

def detect_box(name, resized, lines):
    alignments = align_lines(lines)
    bounds = get_outer_bounds(alignments)
    box = outer_bounds_to_box(bounds)
    
    canvas = np.copy(resized)
    # print_aligned_lines(canvas, alignments)
    # print_bounds(canvas, bounds)
    print_polygon(canvas, box)
    cv2.imshow(name, np.hstack((resized, canvas)))
    

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

def print_aligned_lines(canvas, alignments, width=1):
    for line in alignments[0]:
        cv2.line(canvas, (line[0], line[1]), (line[2], line[3]), (0, 0, 255), width, cv2.LINE_AA)
    for line in alignments[1]:
        cv2.line(canvas, (line[0], line[1]), (line[2], line[3]), (255, 0, 255), width, cv2.LINE_AA)
    for line in alignments[2]:
        cv2.line(canvas, (line[0], line[1]), (line[2], line[3]), (0, 255, 255), width, cv2.LINE_AA)
    for line in alignments[3]:
        cv2.line(canvas, (line[0], line[1]), (line[2], line[3]), (255, 255, 0), width, cv2.LINE_AA)

def get_outer_bounds(alignments):
    top = min(alignments[0], key=lambda e: (e[1] + e[3]) / 2, default=(0,0,0,0))
    right = max(alignments[1], key=lambda e: (e[0] + e[2]) / 2, default=(0,0,0,0))
    bottom = max(alignments[2], key=lambda e: (e[1] + e[3]) / 2, default=(0,0,0,0))
    left = min(alignments[3], key=lambda e: (e[0] + e[2]) / 2, default=(0,0,0,0))
    ret = (top, right, bottom, left)
    return ret

def outer_bounds_to_box(bounds):
    incomplete_box = any(el is (0,0,0,0) for el in bounds)
    if incomplete_box:
        return None
    top_left = get_intersect_helper(bounds[3], bounds[0])
    top_right = get_intersect_helper(bounds[0], bounds[1])
    bottom_right = get_intersect_helper(bounds[1], bounds[2])
    bottom_left = get_intersect_helper(bounds[2], bounds[3])
    return (top_left, top_right, bottom_right, bottom_left)

def print_bounds(canvas, bounds):
    print_aligned_lines(canvas, (
        [bounds[0]],
        [bounds[1]],
        [bounds[2]],
        [bounds[3]],
    ), 3)

def print_polygon(canvas, points):
    points = np.array(points, np.int32)
    cv2.polylines(canvas, [points], True, (0, 255, 255))

def create_empty_canvas():
    return np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), np.uint8)

def get_intersect_helper(a, b):
    return get_intersect((a[0], a[1]), (a[2], a[3]), (b[0], b[1]), (b[2], b[3]))

def get_intersect(a1, a2, b1, b2):
    """ 
    Returns the point of intersection of the lines passing through a2,a1 and b2,b1.
    a1: [x, y] a point on the first line
    a2: [x, y] another point on the first line
    b1: [x, y] a point on the second line
    b2: [x, y] another point on the second line
    """
    s = np.vstack([a1,a2,b1,b2])        # s for stacked
    h = np.hstack((s, np.ones((4, 1)))) # h for homogeneous
    l1 = np.cross(h[0], h[1])           # get first line
    l2 = np.cross(h[2], h[3])           # get second line
    x, y, z = np.cross(l1, l2)          # point of intersection
    if z == 0:                          # lines are parallel
        return (float('inf'), float('inf'))
    return (x/z, y/z)

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
    shape = detect_box(path, resized, lines)

cv2.waitKey()
