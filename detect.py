import math
import cv2
import numpy as np

IMG_WIDTH = 1600
IMG_HEIGHT = 1200
NR_BOTTLES_WIDE = 5
NR_BOTTLES_NARROW = 4
PADDING_WIDE = 0.05
PADDING_NARROW = 0.032
DEFAULT_LINE_WIDTH = 3
BOTTLE_CAP_WIDE_FACTOR = 0.03
BOTTLE_CAP_NARROW_FACTOR = 0.05

PARAM_CANNY_THRESHOLD = 60
PARAM_CANNY_THRESHOLD = 30
PARAM_GAUSSIAN_BLUR = 2.5

def get_small_img(path):
    img = cv2.imread(path)
    return cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT), interpolation = cv2.INTER_CUBIC)
    
def detect_straight_lines(img):
    blurred = cv2.GaussianBlur(img, (0,0), PARAM_GAUSSIAN_BLUR)
    sobelx = np.absolute(cv2.Sobel(blurred,cv2.CV_32F,1,0,ksize=3)) / 255
    sobely = np.absolute(cv2.Sobel(blurred,cv2.CV_32F,0,1,ksize=3)) / 255
    mag_color, ang = cv2.cartToPolar(sobelx, sobely)
    mag_color = mag_color * 255
    mag_color = mag_color.astype(np.uint8)
    mag = mag_color
    mag = cv2.cvtColor(mag, cv2.COLOR_BGR2GRAY)

    traced = cv2.Canny(mag, PARAM_CANNY_THRESHOLD, PARAM_CANNY_THRESHOLD, None, 3)

    lines = cv2.HoughLines(traced, rho = 1, theta = 1*np.pi/180, threshold = 200)
    ret = []
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
            cv2.line(mag_color, pt1, pt2, (0,0,255), DEFAULT_LINE_WIDTH, cv2.LINE_AA)
            ret.append([[pt1[0], pt1[1], pt2[0], pt2[1]]])

    show_pictures(path, [img, mag_color])
    return ret

def detect_box(name, resized, lines):
    canvas = np.copy(resized)
    
    alignments = align_lines(lines)
    # print_aligned_lines(canvas, alignments)
    
    bounds = get_outer_bounds(alignments)
    if (0, 0, 0, 0) in bounds:
        print_bounds(canvas, bounds)
        return None
    
    box = outer_bounds_to_box(bounds)
    
    print_polygon(canvas, box)
    show_pictures(name, [resized, canvas])

    return box
    

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

def print_aligned_lines(canvas, alignments, width=DEFAULT_LINE_WIDTH):
    for line in alignments[0]:
        cv2.line(canvas, (line[0], line[1]), (line[2], line[3]), (0, 0, 255), width, cv2.LINE_AA)
    for line in alignments[1]:
        cv2.line(canvas, (line[0], line[1]), (line[2], line[3]), (255, 0, 255), width, cv2.LINE_AA)
    for line in alignments[2]:
        cv2.line(canvas, (line[0], line[1]), (line[2], line[3]), (0, 255, 255), width, cv2.LINE_AA)
    for line in alignments[3]:
        cv2.line(canvas, (line[0], line[1]), (line[2], line[3]), (255, 255, 0), width, cv2.LINE_AA)

def get_outer_bounds(alignments):
    top = min(alignments[0], key=lambda e: (max(e[1], 0) + max(e[3], 0)) / 2, default=(0,0,0,0))
    right = max(alignments[1], key=lambda e: (max(e[0], 0) + max(e[2], 0)) / 2, default=(0,0,0,0))
    bottom = max(alignments[2], key=lambda e: (max(e[1], 0) + max(e[3], 0)) / 2, default=(0,0,0,0))
    left = min(alignments[3], key=lambda e: (max(e[0], 0) + max(e[2], 0)) / 2, default=(0,0,0,0))
    ret = (top, right, bottom, left)
    return ret

def outer_bounds_to_box(bounds):
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
    ), DEFAULT_LINE_WIDTH)

def print_polygon(canvas, points):
    points = np.array(points, np.int32)
    cv2.polylines(canvas, [points], True, (0, 255, 255), thickness=DEFAULT_LINE_WIDTH)

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


#
#      a0 1 2 3 4b 
#     c+-+-+-+-+-+g       n \in [0;4]
#      +-+-+-+-+-+        a_n = a + (b-a)/5 * (n + 0.5)
#      +-+-+-+-+-+ 
#      +-+-+-+-+-+        n \in [0;3]
#     d+-+-+-+-+-+h       c_n = c + (d-c)/4 * (n + 0.5)
#      e         f
#
def get_cap_positions(name, resized, shape):
    # check which direction is wide and which is narrow
    i_ii = np.subtract(shape[1], shape[0])
    dist_i_ii = np.linalg.norm(i_ii)
    i_iv = np.subtract(shape[3], shape[0])
    dist_i_iv = np.linalg.norm(i_iv)

    # # calculate the other lengths in order to build a scale factor for
    # # the bottle caps
    # iv_iii = np.subtract(shape[2], shape[3])
    # dist_iv_iii = np.linalg.norm(iv_iii)
    # ii_iii = np.subtract(shape[2], shape[1])
    # dist_ii_iii = np.linalg.norm(ii_iii)
    
    if dist_i_ii > dist_i_iv:
        padding_long = PADDING_WIDE * i_ii
        padding_short = PADDING_NARROW * i_iv
        long_1 = (shape[0], shape[1]) # top
        long_2 = (shape[3], shape[2]) # bottom
        short_1 = (shape[0], shape[3]) # left
        short_2 = (shape[1], shape[2]) # right
    else:
        padding_long = PADDING_WIDE * i_iv
        padding_short = PADDING_NARROW * i_ii
        short_1 = (shape[0], shape[1]) # top
        short_2 = (shape[3], shape[2]) # bottom
        long_1 = (shape[0], shape[3]) # left
        long_2 = (shape[1], shape[2]) # right

    # calculate the horizontal and vertical lines
    a = np.asarray(long_1[0]) + padding_long
    b = np.asarray(long_1[1]) - padding_long
    e = np.asarray(long_2[0]) + padding_long
    f = np.asarray(long_2[1]) - padding_long
    c = np.asarray(short_1[0]) + padding_short
    d = np.asarray(short_1[1]) - padding_short
    g = np.asarray(short_2[0]) + padding_short
    h = np.asarray(short_2[1]) - padding_short
    
    canvas = np.copy(resized)
    
    narrow_lines = []
    for n in range(0, NR_BOTTLES_WIDE):
        point_1 = a + (b - a) / NR_BOTTLES_WIDE * (n + 0.5)
        point_2 = e + (f - e) / NR_BOTTLES_WIDE * (n + 0.5)
        narrow_lines.append((point_1, point_2))
        cv2.line(canvas, a2t(point_1), a2t(point_2), (0,0,255), DEFAULT_LINE_WIDTH, cv2.LINE_AA)

    wide_lines = []
    for n in range(0, NR_BOTTLES_NARROW):
        point_1 = c + (d - c) / NR_BOTTLES_NARROW * (n + 0.5)
        point_2 = g + (h - g) / NR_BOTTLES_NARROW * (n + 0.5)
        wide_lines.append((point_1, point_2))
        cv2.line(canvas, a2t(point_1), a2t(point_2), (0,0,255), DEFAULT_LINE_WIDTH, cv2.LINE_AA)


    # found wide and narrow lines, now we need to calculate all
    # intersection point to get the bottle cap positions
    cap_positions = []
    for l1 in narrow_lines:
        for l2 in wide_lines:
            intersection_point = get_intersect(l1[0], l1[1], l2[0], l2[1])
            scale_narrow = get_line_length(l1[0], l1[1]) * BOTTLE_CAP_NARROW_FACTOR
            scale_wide = get_line_length(l2[0], l2[1]) * BOTTLE_CAP_WIDE_FACTOR
            cap_positions.append((intersection_point, scale_narrow, scale_wide))
            cv2.circle(canvas, a2t(intersection_point), int(scale_narrow), (0, 255, 255), thickness=DEFAULT_LINE_WIDTH)
            
    show_pictures(name, [resized, canvas])
    return cap_positions

def a2t(np_array):
    return (int(np_array[0]), int(np_array[1]))

def get_line_length(a, b):
    diff = np.subtract(b, a)
    return np.linalg.norm(diff)

def show_pictures(name, images):
    stack = []
    for i in images:
        img = cv2.resize(i, (400, 300), interpolation = cv2.INTER_CUBIC)
        stack.append(img)
    stacked = np.hstack(stack)
    cv2.imshow(name, stacked)

paths = [
    # './samples/kasten1.jpg',
    # './samples/kasten2.jpg',
    # './samples/kasten3.jpg',
    # './samples/kasten4.jpg',
    # './samples/kasten5.jpg',
    # './samples/kasten6.jpg',
    './samples/example_001.jpg',
    './samples/example_002.jpg',
    './samples/example_003.jpg',
    './samples/example_004.jpg',
    './samples/example_011.jpg',
]

for path in paths:
    print(path)
    resized = get_small_img(path)
    lines = detect_straight_lines(resized)
    shape = detect_box(path, resized, lines)
    if shape == None:
        print('Box not found in picture')
        continue
    cap_positions = get_cap_positions(path, resized, shape)

cv2.waitKey()
