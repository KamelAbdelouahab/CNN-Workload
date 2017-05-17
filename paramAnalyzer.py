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

def specialKernel(nBits,protoFile,modelFile):
	scaleFactor  = 2**(nBits-1) - 1
	netParam     = np.array([])
	cnn          = caffe.Net(protoFile,1,weights=modelFile)

	for p in cnn.params:
		if 'conv' in p:
			layerParam    = cnn.params[p][0].data
			# print p + " " + str(layerParam.shape)
			netParam      = np.append(netParam,layerParam[...].ravel())
	qnetParam  = np.round(scaleFactor * netParam)

	kernelAll  = len(qnetParam)
	kernelZero = np.count_nonzero(qnetParam==0)
	kernelOne  = np.count_nonzero( abs(qnetParam)==1)

	kernelPow2 = np.count_nonzero(abs(qnetParam)==2)
	kernelPow2 += np.count_nonzero(abs(qnetParam)==4)
	kernelPow2 += np.count_nonzero(abs(qnetParam)==8)
	kernelPow2 += np.count_nonzero(abs(qnetParam)==16)
	kernelPow2 += np.count_nonzero(abs(qnetParam)==32)
	kernelPow2 += np.count_nonzero(abs(qnetParam)==64)
	kernelPow2 += np.count_nonzero(abs(qnetParam)==128)
	kernelPow2 += np.count_nonzero(abs(qnetParam)==256)

	return [kernelZero,kernelOne,kernelPow2,kernelAll]

def drawPie(kernelStats,svgFilename='mesouda'):
    data = [kernelStats[0],
            kernelStats[1],
            kernelStats[2],
            kernelStats[3]- kernelStats[0] - kernelStats[1] - kernelStats[2]]
    labels = 'Zero','One','Pow2','etc'
    colors = ['lightskyblue', 'lightcoral', 'yellowgreen','lightgray']
    explode = (0, 0, 0, 0.1)
    plt.pie(data,
            explode=explode,
            autopct='%1.1f%%',
            colors=colors,
            labels=labels,
            startangle=90)
    plt.axis('equal')
    plt.gca().set_position([0, 0, 1, 1])
    plt.show()
    # plt.savefig(svgFilename+".svg")



if __name__ == '__main__':
    if (len(sys.argv) == 3):
        protoFile = sys.argv[1]
        modelFile = sys.argv[2]
        #~ hist,bins  = extractHist(protoFile,modelFile);
        #~ saveHist(hist,bins,modelFile)
    else:
        print (">> Backdoor ! ")
        nBits=6;
        protoFile = '/home/kamel/dev/caffe/models/lenet5/deploy.prototxt'
        modelFile = '/home/kamel/dev/caffe/models/lenet5/qnet_iter_1000.caffemodel'
        kernelStats =  specialKernel(nBits,
									 protoFile,
									 modelFile)
        print kernelStats
        drawPie(kernelStats)
        #~ print (">> paramAnalyser [.prototxt] [.caffemodel]")
