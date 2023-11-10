import cv2
import numpy as np

def evaluate(img1,img2,l):
    # if RGB, then convert into grayscale
    if len(img1.shape) == 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    if len(img2.shape) == 3:
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl1 = clahe.apply(img1)
    cl2 = clahe.apply(img2)

    cl1 = np.where((cl1 > 200) | (cl1 < 5), 0, cl1)
    cl2 = np.where((cl2 > 230) | (cl2 < 5), 0, cl2)

    cv2.imwrite(f'cl1{l}.jpg',cl1)
    cv2.imwrite(f'cl2{l}.jpg', cl2)
    # s1: intersection,s2: union
    s1 = np.count_nonzero(cl1*cl2)
    s2 = min(np.count_nonzero(cl1+cl1*cl2), np.count_nonzero(cl2+cl1*cl2))
    return s1,s2
