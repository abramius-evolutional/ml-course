import sys
sys.path.append('/usr/local/lib/python3.5/site-packages')
from sklearn.cluster import KMeans
from skimage.io import imread
from skimage import img_as_float
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pylab

image = imread('parrots.jpg')

im = img_as_float(image)
# pylab.imshow(im)
# pylab.show()

features = im.reshape([im.shape[0]*im.shape[1],im.shape[2]])

def log10(x):
    return np.log(x)/np.log(10)

for n_clusters in [11,]:
    features_mean = np.copy(features)
    features_median = np.copy(features)

    kmeans = KMeans(init='k-means++', random_state=241, n_clusters=n_clusters)
    kmeans.fit(features)
    labels = set(kmeans.labels_)

    for label in labels:
        ids = np.where(kmeans.labels_==label)
        interest = features[ids]
        mean_color = interest.mean(axis=0)
        median_color = np.median(interest, axis=0)
        features_mean[ids] = mean_color
        features_median[ids] = median_color

    mean_im = features_mean.reshape(im.shape)
    median_im = features_median.reshape(im.shape)
    # pylab.imshow(mean_im)
    # pylab.show()

    def PSNR(im1, im2):
        n = im1.shape[0]*im1.shape[1]*im1.shape[2]
        I = im1.reshape(n)
        K = im2.reshape(n)
        RMSE = np.sqrt(np.sum((I - K)**2) / float(n))
        return 20.*log10(np.max(I)/RMSE)

    print('n_clusters =', n_clusters)
    print('\t', PSNR(im, mean_im))
    print('\t', PSNR(im, median_im))

pylab.imshow(mean_im)
pylab.show()