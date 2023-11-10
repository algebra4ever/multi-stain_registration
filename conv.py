import numpy as np

def cov2D(img,stride):
    '''
    :param img: input image, ndarray
    :param stride: kernel size, int
    :return: the density of valid pixels of input img
    '''

    #calculate the valid pixel density of input image using image convolution
    m, n = img.shape
    kernel = np.ones((stride,stride))
    s = np.zeros((int(m/stride),int(n/stride)))
    for i in range(int(m/stride)):
        for j in range(int(n/stride)):
            s[i,j] = np.sum(kernel*img[i*stride:(i+1)*stride,j*stride:(j+1)*stride])
    return s



