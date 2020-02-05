import math
import cv2
import numpy as np

# adapted from https://stackoverflow.com/a/26613004/2752278

def remove_barrel(src, distortion, focal_length):
    width  = src.shape[1]
    height = src.shape[0]

    distCoeff = np.zeros((4,1),np.float64)

    k1 = distortion; # negative to remove barrel distortion
    k2 = 0.0;
    p1 = 0.0;
    p2 = 0.0;

    distCoeff[0,0] = k1;
    distCoeff[1,0] = k2;
    distCoeff[2,0] = p1;
    distCoeff[3,0] = p2;

    # assume unit matrix for camera
    cam = np.eye(3,dtype=np.float32)

    cam[0,2] = width/2.0  # define center x
    cam[1,2] = height/2.0 # define center y
    cam[0,0] = focal_length        # define focal length x
    cam[1,1] = focal_length        # define focal length y

    return cv2.undistort(src,cam,distCoeff)    
  
if __name__ == '__main__':
    src = cv2.imread("./samples/file-2020-02-05 11:36:07.632.jpg")
    dst = remove_barrel(src, -1.0e-4, 8.)
    cv2.imshow('dst', dst)
    cv2.waitKey(0)
