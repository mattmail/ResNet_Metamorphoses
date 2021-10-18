import numpy as np
import matplotlib.pyplot as plt
import os


i = 0
for image in os.listdir('../data_miccai_2D_2021'):
    if image[:5] == "BraTS":
        if i >= 0:
            im = np.transpose(np.load('../data_miccai_2D_2021/' + image + "/" + image + "_t1.npy"))
            plt.imshow(im, cmap='gray')
            plt.title(image)
            plt.show()
            if i >= 40:
                break
        i+=1
