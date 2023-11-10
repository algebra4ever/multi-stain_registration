from sklearn.cluster import KMeans
import numpy as np
from conv import cov2D
import cv2

def seg(img,k,stride):
    '''
    :param img: initial image(RGB)
    :param k: k for K-means
    :param stride: stride from conv
    :return: clusters' labels and centers
    '''
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred_img = cv2.GaussianBlur(gray_img, (3, 3), sigmaX=1)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(blurred_img)
    # edge detection
    edge_img = cv2.Canny(cl, 100, 200)
    cv2.imwrite('edge.jpg',edge_img)
    s = cov2D(edge_img, stride) / 255
    points = np.argwhere(s > 10)
    img_new = np.zeros_like(edge_img)
    for point in points:
        img_new[point[0] * stride:(point[0] + 1) * stride, point[1] * stride:(point[1] + 1) * stride] \
            = gray_img[point[0] * stride:(point[0] + 1) * stride, point[1] * stride:(point[1] + 1) * stride]

    cluster_points = np.argwhere(img_new) #all the points used for clustering
    kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
    labels = kmeans.fit_predict(cluster_points)
    clu_centers = kmeans.cluster_centers_ #cluster centers
    return labels,cluster_points,clu_centers


#visualization of segmentation result
# def seg_visual(labels,cluster_points,img):
#     image_color = np.zeros_like(img, dtype=np.uint8)
#     # Assign colors to each cluster
#     colors = np.random.randint(0, 255, size=(np.max(labels) + 1, 3), dtype=np.uint8)
#     for point, label in list(zip(cluster_points, labels)):
#         image_color[point[0], point[1]] = colors[label]
#     cv2.imshow('Clustered Edges', image_color)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()