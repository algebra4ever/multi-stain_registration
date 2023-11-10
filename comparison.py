import cv2
import numpy as np
from linear import sub_register_linear
from evaluate import evaluate
from segmentation import seg
from linear import affine
from main import img1,img2,k,img_affine


gray_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)


labels1,cluster_points1,centers1 = seg(img1,k,8)
labels2,cluster_points2,centers2 = seg(img2,k,8)
kk = np.zeros((k,2))
kk[:,0] = np.arange(k)
dist = np.zeros((k,k))
for i in range(k):
    for j in range(k):
        dist[i, j] = np.linalg.norm(centers1[i,:]-centers2[j,:],2)#计算各聚类中心两两之间的欧氏距离
kk[:,1] = np.argmin(dist,axis=1)

aligned_img = []
img2_r = []

s1 = []
s2 = []
s3 = []
s4 = []
s5 = []
s6 = []
overlap = np.zeros(k)
M1 = []
for l in range(k):
    aligned_img_l,img2_r_l,M1_l = sub_register_linear(img1, img2, kk, l, cluster_points1, cluster_points2, labels1, labels2,alpha=0.8)
    aligned_img.append(aligned_img_l)
    img2_r.append(img2_r_l)
    M1.append(M1_l) # to obtain the transform matrix
    s1_l, s2_l = evaluate(aligned_img_l, img2_r_l, l)
    dst = cv2.addWeighted(aligned_img_l, 0.5, img2_r_l, 0.5, 0)
    cv2.imwrite(f'step1+2_{l}.jpg', dst)
    s1.append(s1_l)
    s2.append(s2_l)
    overlap[l] = s1[l] / s2[l]
    avg_overlap_12 = np.sum(s1) / np.sum(s2) # step 1+2

    img_bun = cv2.imread(f'bun1{l}.jpg') # registered sub-image using ImageJ
    s1_bun, s2_bun = evaluate(img_bun, img2_r_l, l)
    s3.append(s1_bun)
    s4.append(s2_bun)
    avg_overlap_13 = np.sum(s3) / np.sum(s4) # step 1+3

    Registered_Target_Image = cv2.imread(f'Registered Target Image1{l}.jpg')
    s5_l,s6_l = evaluate(Registered_Target_Image, img2_r_l,l)
    dst = cv2.addWeighted(Registered_Target_Image, 0.5, img2_r_l, 0.5, 0)
    # 显示融合后的图像
    cv2.imwrite(f'final{l}.jpg', dst)
    s5.append(s5_l)
    s6.append(s6_l)
    overlap[l] = s5[l]/s6[l]
    avg_overlap = np.sum(s5)/np.sum(s6) # step 1+2+3

img_23 = cv2.imread('Registered Target Image2+3.jpg')
dst1 = cv2.addWeighted(img_23, 0.5, img2, 0.5, 0)
cv2.imwrite(f'final.jpg', dst1)
overlap_23 = evaluate(img_23,img2,l=k+2)
avg_overlap_23 = overlap_23[0]/overlap_23[1] # step 2+3

overlap_affine = evaluate(img_affine,img2,l=k+3)
avg_overlap_affine = overlap_affine[0]/overlap_affine[1]

img_bunwarpj=cv2.imread('Registered Target Image bunwarpj.jpg')
overlap_b = evaluate(img_bunwarpj,img2,l=k+4)
avg_overlap_b = overlap_b[0]/overlap_b[1]