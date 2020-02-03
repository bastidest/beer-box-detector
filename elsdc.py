from ctypes import *
import numpy as np
import cv2
import re

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

class ELSDcWrapper:
    def __init__(self):
        self.libELSDc = cdll.LoadLibrary("./libELSDc")
        self.libELSDc.new_PImageDouble_ini.restype = POINTER(ImageDouble)
        self.libELSDc.new_PImageInt_ini.restype = POINTER(ImageInt)
    
    def execute(self, width, height, input_img):
        T_VOID_P = POINTER(None)
        
        self.in_img = self.libELSDc.new_PImageDouble_ini(c_uint(width), c_uint(height), c_double(0.0))
        self.out_img = self.libELSDc.new_PImageInt_ini(c_uint(width), c_uint(height), c_int(0))

        for row in range(0, height):
            for col in range(0, width):
                self.in_img[0].data[row * width + col] = input_img[row][col]
                        
        self.ell_count = c_int(0)
        self.ell_out = T_VOID_P()
        self.ell_labels = POINTER(c_int)()

        self.poly_count = c_int(0)
        self.poly_out = T_VOID_P()
        self.poly_labels = POINTER(c_int)()

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

        ret = np.zeros((test_h, test_w))
        
        for row in range(0, height):
            for col in range(0, width):
                 ret[row][col] = self.out_img[0].data[row * width + col]
                
        return ret

test_w = 400
test_h = 300

test = ELSDcWrapper()
img = cv2.imread('./samples/example_005.jpg')
img = cv2.resize(img, (test_w, test_h), interpolation = cv2.INTER_CUBIC)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow('before', gray)
returned = test.execute(test_w, test_h, gray)
cv2.imshow('after', returned)
cv2.waitKey()

