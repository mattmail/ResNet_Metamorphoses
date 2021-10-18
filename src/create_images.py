import numpy as np
import matplotlib.pyplot as plt
import elasticdeform
import cv2
from scipy import ndimage
import os
import matplotlib
from random import random

target = cv2.resize(cv2.imread("reg_test_m0.png", cv2.IMREAD_GRAYSCALE), (200,200))
if not os.path.exists('../images'):
    os.mkdir('../images')


for i in range(1000):
    print("creating image %d" % i)
    """angle = np.random.randint(-20, 20)"""
    if random() > 0.7:
        points = 2
    else:
        points = 3
    im_deformed = elasticdeform.deform_random_grid(target, order=1, sigma=8, points=points)

    """rectangle = np.zeros((target.shape))
    h, w = rectangle.shape
    width = np.random.randint(5, 15)
    rectangle[h//2-width:h//2+width, 0:w//2] = 1
    rectangle = ndimage.rotate(rectangle, angle, reshape=False, order=0)
    rectangle = (rectangle > .5) * 1
    #rectangle[rectangle > .5] = 1

    segmentation = ((rectangle + im_deformed) == 256)*1
    im_deformed[rectangle==1] = 0."""

    """if not os.path.exists('../images_varying_cut/image_%d' % i):
        os.mkdir('../images_varying_cut/image_%d' % i)"""

    matplotlib.image.imsave("../images/source_%d.png" % i, im_deformed/255, cmap="gray")
    #matplotlib.image.imsave("../images_varying_cut/image_%d/segmentation.png" % i, segmentation, cmap="gray")

target = cv2.resize(cv2.imread("reg_test_m0t.png", cv2.IMREAD_GRAYSCALE), (200,200))
for i in range(1000, 2000):
    print("creating image %d" % i)
    angle = np.random.randint(-20, 20)
    if random() > 0.8:
        points = 2
    else:
        points = 3
    im_deformed = elasticdeform.deform_random_grid(target, order=1, sigma=8, points=points)

    """rectangle = np.zeros((target.shape))
    h, w = rectangle.shape
    width = np.random.randint(5, 15)
    rectangle[h // 2 - width:h // 2 + width, 0:w // 2] = 1
    rectangle = ndimage.rotate(rectangle, angle, reshape=False, order=0)
    rectangle = (rectangle > .5) * 1
    #rectangle[rectangle > .5] = 1

    segmentation = ((rectangle + im_deformed) == 256)*1
    im_deformed[rectangle==1] = 0."""

    """if not os.path.exists('../images_varying_cut/image_%d' % i):
        os.mkdir('../images_varying_cut/image_%d' % i)"""

    matplotlib.image.imsave("../images/source_%d.png" % i, im_deformed/255, cmap="gray")
    #matplotlib.image.imsave("../images_rayures/image_%d/segmentation.png" % i, segmentation, cmap="gray")


"""target = cv2.resize(cv2.imread("reg_test_m0t.png", cv2.IMREAD_GRAYSCALE), (200,200))
if not os.path.exists('images'):
    os.mkdir('images')

for i in range(250):
    print("creating image %d" % (250+i))
    if random() > 0.2:
        points = 2
    else:
        points = 3
    im_deformed = elasticdeform.deform_random_grid(target, order=1, sigma=8, points=points)

    matplotlib.image.imsave("images_rayures/source_%d.png" % (i+250), im_deformed/255, cmap="gray")"""


"""image_list = os.listdir('images')
n_im = len(image_list)

for i in range(n_im):
    print("Comparing", image_list[i])
    for j in range(i,n_im):
        im1 = cv2.imread('images/'+image_list[i])/255
        im2 = cv2.imread('images/'+image_list[j])/255

        if np.sum((im1 - im2)**2 )< 5000 and i!=j:
            print("these two images are similar")
            print(image_list[i])
            print(image_list[j])
            plt.imshow(im1, cmap="gray")
            plt.show()
            plt.imshow(im2, cmap="gray")
            plt.show()"""
