import cv2
import numpy as np
import detect

images = [
    # './samples/kasten1.jpg',
    # './samples/kasten2.jpg',
    # './samples/kasten3.jpg',
    # './samples/kasten4.jpg',
    # './samples/kasten5.jpg',
    # './samples/kasten6.jpg',
    ('./samples/example_001.jpg',((34,24),(264,27),(264,197),(33,199))),
    ('./samples/example_002.jpg',((34,24),(264,27),(264,197),(33,199))),
    ('./samples/example_003.jpg',((35,25),(264,27),(263,197),(33,199))),
    ('./samples/example_004.jpg',((35,25),(264,27),(263,197),(33,199))),
    ('./samples/example_011.jpg',((35,25),(264,27),(263,197),(33,199))),
]

def compare_boxes(expected, actual):
    if actual == None:
        print('Box not found in picture')
        return float('nan')

    print(expected)
    # round actual
    actual = tuple(map(lambda xy: (round(xy[0]),round(xy[1])), actual))
    print(actual)

    comp_sum = 0
    for i in range(0,4):
        comp =  (expected[i][0]-actual[i][0])**2
        comp += (expected[i][1]-actual[i][1])**2
        print(comp)
        comp_sum += comp
    comp = comp_sum/4
    print('result compare:', comp)
    return comp

# adapt box coordinates to image width and height
def adapt_box(path, box):
    img = cv2.imread(path)
    height, width, channels = img.shape
    print('actual image size: {}x{}'.format(width, height))

    new_box = ()
    resizer = lambda xy: (xy[0]*detect.IMG_WIDTH/width, xy[1]*detect.IMG_HEIGHT/height)
    for point in box:
        new_point = resizer(point)
        new_box += (new_point,)
    return new_box

test_results = []
for image in images:
    path, expected_box = image
    print(path)
    expected_box = adapt_box(path, expected_box)
    resized = detect.get_small_img(path)
    lines = detect.detect_straight_lines(resized)
    shape = detect.detect_box(path, resized, lines)
    res = compare_boxes(expected_box, shape)

    test_result = (path, res)
    test_results.append(test_result)

    if shape == None:
        continue

    cap_positions = detect.get_cap_positions(path, resized, shape)

print('######################################')
sorted_test_results = sorted(test_results, key = lambda t: float('inf') if np.isnan(t[1]) else t[1])
for res in sorted_test_results:
    print(res)
