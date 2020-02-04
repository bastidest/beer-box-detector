from ctypes import *
import numpy as np
import cv2
import math

class ImageInt(Structure):
    _fields_ = [
        ('data', POINTER(c_int)),
        ('xsize', c_int),
        ('ysize', c_int),
    ]

class ImageDouble(Structure):
    _fields_ = [
        ('data', POINTER(c_double)),
        ('xsize', c_int),
        ('ysize', c_int),
    ]

class Ring(Structure):
    _fields_ = [
        ('x1', c_double),          # start and endpoints of the ellipse
        ('y1', c_double),
        ('x2', c_double),
        ('y2', c_double),
        ('width', c_double),       # width of ellipse
        ('cx', c_double),          # ellipse center
        ('cy', c_double),
        ('theta', c_double),       # ellipse orientation, 0 => circle
        ('ax', c_double),          # ellipse axes ax == bx => circle
        ('bx', c_double),
        ('ang_start', c_double),   # angles
        ('ang_end', c_double),
        ('wmin', c_double),        # width towards interior and exterior
        ('wmax', c_double),
        ('full', c_int),           # complete circle?
    ]

class PointD(Structure):
    _fields_ = [
        ('x', c_double),
        ('y', c_double),
    ]

class Polygon(Structure):
    _fields_= [
        ('dim', c_int),
        ('pts', POINTER(PointD)),
    ]

class Ellipse:
    def __init__(self, center, width, height, degrees, thickness):
        self.center = center
        self.width = width
        self.height = height
        self.degrees = degrees
        self.thickness = thickness

    def __repr__(self):
        return str(self.__dict__)

class ELSDcWrapper:
    def __init__(self):
        self.libELSDc = cdll.LoadLibrary("./libELSDc")
        self.libELSDc.new_PImageDouble_ini.restype = POINTER(ImageDouble)
        self.libELSDc.new_PImageInt_ini.restype = POINTER(ImageInt)
    
    def execute(self, width, height, input_img):
        T_VOID_P = POINTER(None)

        # allocate input and output arrays on the heap
        self.in_img = self.libELSDc.new_PImageDouble_ini(c_uint(width), c_uint(height), c_double(0.0))
        self.out_img = self.libELSDc.new_PImageInt_ini(c_uint(width), c_uint(height), c_int(0))

        # copy the np input array to the allocated C array
        for row in range(0, height):
            for col in range(0, width):
                self.in_img[0].data[row * width + col] = input_img[row][col]
                        
        # allocate output parameters
        self.ell_count = c_int(0)
        self.ell_out = POINTER(Ring)()
        self.ell_labels = POINTER(c_int)()

        self.poly_count = c_int(0)
        self.poly_out = POINTER(Polygon)()
        self.poly_labels = POINTER(c_int)()

        # call the actual ELSDc algorithm
        self.libELSDc.ELSDc(
            self.in_img,
            pointer(self.ell_count),
            pointer(self.ell_out),
            pointer(self.ell_labels),
            pointer(self.poly_count),
            pointer(self.poly_out),
            pointer(self.poly_labels),
            self.out_img,
        )

        # print("found %d ellipses and %d polygons" % (self.ell_count.value, self.poly_count.value))

        # copy the C array back to a np array
        ret = np.zeros((height, width))
        for row in range(0, height):
            for col in range(0, width):
                 ret[row][col] = self.out_img[0].data[row * width + col]

        # extract the ellipsis data
        ellipses = []
        for i in range(0, self.ell_count.value):
            e = self.ell_out[i]
            center = (e.cx, e.cy)
            width = e.ax
            height = e.bx
            diff = e.ang_start - e.ang_end
            if diff < 0:
                diff += 2 * math.pi
            thickness = e.width
            ellipses.append(Ellipse(center, width, height, diff, thickness))

        return ret, ellipses

def a2t(np_array):
    return (int(np_array[0]), int(np_array[1]))

if __name__ == '__main__':
    test_w = 400
    test_h = 300

    test = ELSDcWrapper()
    img = cv2.imread('./samples/example_005.jpg')
    img = cv2.resize(img, (test_w, test_h), interpolation = cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cv2.imshow('before', gray)
    returned, ellipses = test.execute(test_w, test_h, gray)
    returned = returned * 255
    returned = returned.astype(np.uint8)
    returned = cv2.cvtColor(returned, cv2.COLOR_GRAY2BGR)
    for e in ellipses:
        stretch_factor = abs(1 - (e.width / e.height))
        if stretch_factor > 0.2:
            continue
        if e.width > 16 or e.height > 16:
            continue
        cv2.circle(returned, a2t(e.center), int(e.height), (255, 0, 255), thickness=int(e.thickness * 10))

    cv2.imshow('after', returned)
    cv2.waitKey()

