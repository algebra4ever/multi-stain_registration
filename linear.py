import cv2
import numpy as np


sift = cv2.SIFT_create()
def sub_register_linear(img1,img2,kk,l,cluster_points1,cluster_points2,labels1,labels2,alpha):

    t1 = cluster_points1[np.where(labels1==l)]
    t2 = cluster_points2[np.where(labels2==int(kk[np.where(kk[:, 0] == l),1]))]

    img1_r = img1[min(t1[:,0]):(max(t1[:, 0])+1),min(t1[:,1]):(max(t1[:, 1])+1)]
    img2_r = img2[min(t2[:, 0]):(max(t2[:, 0]) + 1), min(t2[:, 1]):(max(t2[:, 1]) + 1)]

    aligned_img,M = affine(img1_r,img2_r,l,alpha)
    # segmentation result
    cv2.imwrite(f'img1{l}.jpg', img1_r)
    cv2.imwrite(f'img2{l}.jpg', img2_r)
    return aligned_img,img2_r,M

def affine(img1,img2,l,alpha = 0.8):
    gray_imgr1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray_imgr2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    # Adaptive histogram equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl1 = clahe.apply(gray_imgr1)
    cl2 = clahe.apply(gray_imgr2)
    kp1, des1 = sift.detectAndCompute(cl1, None)
    kp2, des2 = sift.detectAndCompute(cl2, None)

    # FLANN
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    # match key points
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < alpha * n.distance:
            good_matches.append(m)

    # extract the location of key points
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # using RANSAC to estimate the affine matrix
    M, mask = cv2.estimateAffine2D(src_pts, dst_pts, cv2.RANSAC)
    aligned_img = cv2.warpAffine(img1, M, (img2.shape[1], img2.shape[0]))

    # fusion image
    dst = cv2.addWeighted(aligned_img, 0.5, img2, 0.5, 0)
    cv2.imwrite(f'{l}.jpg', dst)
    # image after affine transformation
    cv2.imwrite(f'affine_original_{l}.jpg', aligned_img)
    aligned_img = np.where(aligned_img == 0, 255, aligned_img)
    cv2.imwrite(f'affine_{l}.jpg', aligned_img)
    return aligned_img,M


