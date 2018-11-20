# -*- coding: UTF-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
params = {
    'grid.color' : 'k',
    'grid.linestyle': 'dashdot',
    'grid.linewidth': 0.6,
    'font.family': 'Linux Biolinum O',
    'axes.labelsize': 8,
    'font.size': 8,
    'legend.fontsize': 15,
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,
    'axes.facecolor' : 'white'
   }
rcParams.update(params)


width = 0.5

# Classifiers
Model = ['AlexNet', 'Overfeat', 'GoogleNet', 'VGG19', 'ResNet152']
year  = ['2012', '2013', '2014', '', '2015']
nets  = np.array([1, 2 ,3 , 3.5, 4.5])
Top1  = 100 - np.array([42.90, 33.96, 31.30,  27.3, 23.0])
Top5  = 100 - np.array([19.80, 13.24, 10.07,  9.00, 6.70])

plt.figure(figsize=(7, 5))
plt.bar(nets-0.5*width, Top5, width, color='g')
plt.bar(nets-0.5*width, Top1, width, color='b')
plt.text(0.5, 82, Model[0], fontsize=15)
plt.text(1.4, 88, Model[1], fontsize=15)
plt.text(2.2, 91, Model[2], fontsize=15)
plt.text(2.8, 93, Model[3], fontsize=15)
plt.text(3.8, 95, Model[4], fontsize=15)
plt.axis([0,6, 50, 100])
plt.xticks(nets, year)
plt.legend(['Top5 Acc.', 'Top1 Acc.'], bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
plt.grid()
plt.savefig("ClassifiersComp.pdf", bbox_inches ='tight')
plt.show()

# Detectors
# Model = ['Fast R-CNN\\AlexNet', 'Fast R-CNN\\VGG16', 'Fast-RCNN\\ResNet', 'YOLOv1', 'YOLOv2','YOLOv3']
# year  = ['2015', '', '', '2016', '2017', '2018']
# nets  = np.array([1, 1.5 ,2 , 3, 4, 5])
# mAP50 = 100 - np.array([42.90, 33.96, 31.30,  27.3, 23.0])
