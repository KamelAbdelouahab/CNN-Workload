# -*- coding: UTF-8 -*-

import sys
import os
import io
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.font_manager import _rebuild; _rebuild()
plt.rcParams['grid.color'] = 'k'
plt.rcParams['grid.linestyle'] = 'dashdot'
plt.rcParams['grid.linewidth'] = 0.4
plt.rcParams['font.family'] = 'Garamond'

try:
    CAFFE_ROOT = os.environ['CAFFE_ROOT']
    CAFFE_PYTHON_LIB = CAFFE_ROOT + '/python'
    sys.path.insert(0, CAFFE_PYTHON_LIB)
except KeyError:
    print("Warning: CAFFE_ROOT environment variable not set")
os.environ['GLOG_minloglevel'] = '2'  # Supresses Display on console
import caffe

width = 0.2
layer_names = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'conv6',
               'conv7', 'conv8', 'conv9', 'conv10', 'conv11', 'conv12', 'conv13']
y_names = ['$10^{-2}$', '$10^{-1}$', '$1$', '$10^1$', '$10^2$']
layers = np.linspace(1, 13, 13, endpoint=True)
# C2C = CJK / (CHW + CJK + NUV)


alexnet    = [0.0782, 1.3787, 8.0093, 6.6977, 5.7124, 0, 0, 0, 0, 0, 0, 0, 0]
vgg16      = [0.0005, 0.0057, 0.0306, 0.0459, 0.2447, 0.3668,
              0.3668, 1.9517, 2.9220, 2.9220, 11.4913, 11.4913, 11.4913]
darknet    = [0.0003, 0.0059, 0.0468, 0.3739, 2.9653,
              22.9254, 161.6842, 30.6513, 0, 0, 0, 0, 0]
yolov3tiny = [0.0100, 0.0022, 0.0177, 0.1419, 1.1311,
              8.9302, 59.0769, 4.1124, 29.5385, 3.4272, 0, 0, 0]

log_alexnet = np.log10((alexnet))
log_vgg16 = np.log10((vgg16))
log_darknet = np.log10((darknet))
log_yolov3tiny = np.log10((yolov3tiny))

plt.figure(figsize=(12, 3))
plt.gca().grid(axis='y')
plt.bar(layers + 0.5 * width, log_alexnet, width)
plt.bar(layers - 0.5 * width, log_vgg16, width)
plt.bar(layers + 1.5 * width, log_darknet, width)
plt.bar(layers - 1.5 * width, log_yolov3tiny, width)
plt.xticks(layers, layer_names)
plt.yticks([-2, -1, 0, 1, 2], y_names)
plt.legend(['Alexnet', 'VGG16', 'Darknet', 'YOLOv3-tiny'])
plt.ylabel("CTC Ratio $MACs/MemAccess$")
plt.show()


true_alexnet    = np.array(alexnet)
true_vgg16      = np.array(vgg16)
true_darknet    = np.array(darknet)
true_yolov3tiny = np.array(yolov3tiny)

plt.figure(figsize=(12, 3))
plt.grid()
plt.bar(layers + 0.5 * width, true_alexnet, width)
plt.bar(layers - 0.5 * width, true_vgg16, width)
plt.bar(layers + 1.5 * width, true_darknet, width)
plt.bar(layers - 1.5 * width, true_yolov3tiny, width)
plt.xticks(layers, layer_names)
plt.yscale('log', basey=10)
plt.legend(['Alexnet', 'VGG16', 'Darknet', 'YOLOv3-tiny'])
plt.ylabel("CTC Ratio $MACs/MemAccess$")
plt.show()
