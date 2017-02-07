# Kamel ABDELOUAHAB
# DREAM - Institut Pascal

import sys
import os
import io
import numpy as np
import math
import matplotlib.pyplot as plt
import pylab as P

HOME                = os.environ['HOME']
CAFFE_DIRNAME       = HOME + '/caffe'
CAFFE_PYTHON_LIB    = CAFFE_DIRNAME+'/python'
sys.path.insert(0, CAFFE_PYTHON_LIB)
import caffe;

def extractHist(protoFile, modelFile):
    cnn          = caffe.Net(protoFile,1,weights=modelFile)
    netParam     = cnn.params
    netHist      = np.array([])
    bins         = np.array([])
    netAllParam  = np.array([])
    # Print CNN shape
    print ">> Network parameter shape"
    for p in netParam:
        if 'conv' in p:
            layerParam    = netParam[p][0].data
            print p + " " + str(layerParam.shape)
            netAllParam   = np.append(netAllParam,layerParam[...].ravel())
            netHist,bins  = np.histogram(netAllParam[...].ravel(),bins=200)
    return netHist,bins

def saveHist(hist,bins,name):
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    fig, ax = plt.subplots()
    ax.bar(center, hist, align='center', width=width)
    name =  name + ".png"
    fig.savefig(name)

if __name__ == '__main__':
    if (len(sys.argv) == 3):
        protoFile = sys.argv[1]
        modelFile = sys.argv[2]
        hist,bins  = extractHist(protoFile,modelFile);
        saveHist(hist,bins,modelFile)
    else:
        print (">> Not enoght arguments ! ")
        print (">> paramAnalyser [.prototxt] [.caffemodel]")
