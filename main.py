import cv2
import numpy as np

from segmentation import seg
#from segmentation import seg_visual
from linear import sub_register_linear
from evaluate import evaluate
from linear import affine

# read image
img1 = cv2.imread('HE.jpg')
img2 = cv2.imread('ER.jpg')

#contrast enhancement if needed
# img1 = np.uint8(np.clip((cv2.add(1.2*img1,0)), 0, 255))
# img2 = np.uint8(np.clip((cv2.add(1.2*img2,0)), 0, 255))
# cv2.imwrite('img1.jpg',img1)
# cv2.imwrite('img2.jpg',img2)

gray_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# def determine_k(img1,img2):
#     for it in range(1,10):
#         labels1, cluster_points1, centers1 = seg(img1, it, 8)
#         labels2, cluster_points2, centers2 = seg(img2, it, 8)
#         kk = np.zeros((it, 2))
#         kk[:, 0] = np.arange(it)
#         dist = np.zeros((it,it))
#         for i in range(it):
#             for j in range(it):
#                 dist[i, j] = np.linalg.norm(centers1[i, :] - centers2[j, :], 2)
#         kk[:, 1] = np.argmin(dist, axis=1)
#         aligned_img = []
#         img2_r = []
#         s1 = []
#         s2 = []
#         avg_overlap_12 = []
#         overlap = np.zeros(it)
#         M1 = []
#         for l in range(it):
#             aligned_img_l, img2_r_l, M1_l = sub_register_linear(img1, img2, kk, l, cluster_points1, cluster_points2, labels1,labels2, alpha=0.8)
#             aligned_img.append(aligned_img_l)
#             img2_r.append(img2_r_l)
#             M1.append(M1_l)  # to obtain the transform matrix
#             s1_l, s2_l = evaluate(aligned_img_l, img2_r_l, l)
#             dst = cv2.addWeighted(aligned_img_l, 0.5, img2_r_l, 0.5, 0)
#             cv2.imwrite(f'step1+2_{l}.jpg', dst)
#             s1.append(s1_l)
#             s2.append(s2_l)
#             overlap[l] = s1[l] / s2[l]
#         avg_overlap_12[it] = np.sum(s1) / np.sum(s2)
#
#     k = np.argmax(avg_overlap_12)
#     return k

# k = determine_k(img1,img2)
k = 2

labels1,cluster_points1,centers1 = seg(img1,k,8)
labels2,cluster_points2,centers2 = seg(img2,k,8)

# visualization of segmentation result
#seg_visual(labels1,cluster_points1,img1)
#seg_visual(labels2,cluster_points2,img2)


# register sub-images
kk = np.zeros((k,2))
kk[:,0] = np.arange(k)
dist = np.zeros((k,k))
for i in range(k):
    for j in range(k):
        # Eulidean distance between each cluster center under same coordinate system
        dist[i, j] = np.linalg.norm(centers1[i,:]-centers2[j,:],2)
kk[:,1] = np.argmin(dist,axis=1)


aligned_img = []
img2_r = []

s1 = []
s2 = []
overlap = np.zeros(k)
M1 = []
for l in range(k):
    aligned_img_l,img2_r_l,M1_l = sub_register_linear(img1, img2, kk, l, cluster_points1, cluster_points2, labels1, labels2,alpha=0.8)
    aligned_img.append(aligned_img_l)
    img2_r.append(img2_r_l)
    M1.append(M1_l) # to obtain the transform matrix

img_affine,_ = affine(img1,img2,l=k+1,alpha=0.8)

