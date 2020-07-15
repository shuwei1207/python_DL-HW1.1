# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 16:24:09 2020

@author: SeasonTaiInOTA
"""

import matplotlib.pyplot as plt
import numpy as np

#標準化
#x_test = x_test.astype('float32') / 255.
#x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

#latent distribution
x_test_encoded = network.predict(x_test)

plt.figure(figsize=(6, 6))
plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c= y_20 , s= 10)
#plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c= y_80 , s= 10)
plt.colorbar()
plt.show()