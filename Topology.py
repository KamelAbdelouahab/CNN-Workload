#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys
import os
import io
import numpy as np
import math
import matplotlib.pyplot as plt
import pylab as P

CAFFE_ROOT       = os.environ['CAFFE_ROOT']
CAFFE_PYTHON_LIB = CAFFE_ROOT+'/python'
sys.path.insert(0, CAFFE_PYTHON_LIB)
os.environ['GLOG_minloglevel'] = '2' # Supresses Display on console
import caffe;


def Topology(protoFile, modelFile):
    cnn     = caffe.Net(protoFile,1,weights=modelFile)
    params  = cnn.params
    blobs   = cnn.blobs
    
    for l in cnn._layer_names:
        layerId = list(cnn._layer_names).index(l)
        layerType =  cnn.layers[layerId].type

        if (layerType == 'Convolution'):
            non_lin = ""        
            N = params[l][0].data.shape[0]
            C = params[l][0].data.shape[1]
            J = params[l][0].data.shape[2]
            K = params[l][0].data.shape[3]
            U = blobs[l].data.shape[2]
            V = blobs[l].data.shape[3]
            for i in range (4):
                if (cnn.layers[layerId+i].type == 'ReLU'):
                    non_lin = non_lin + " " + "ReLU"
                if (cnn.layers[layerId+i].type == 'TanH'):
                    non_lin = non_lin + " " + "TanH"
                if (cnn.layers[layerId+i].type == 'Sigmoid'):
                    non_lin = non_lin + " " + "Sigmoid"
                if (cnn.layers[layerId+i].type == 'Pooling'):
                    non_lin = non_lin + " " + "Pooling"
                if (cnn.layers[layerId+i].type == 'LRN' or cnn.layers[layerId+i].type == "BatchNorm"):
                    non_lin = non_lin + " " + "BN"        
            print(l      + "\t" + 
                  str(C) + "\t" + 
                  str(N) + "\t" +  
                  str(J) + "\t" +  
                  str(K) + "\t" +  
                  str(U) + "\t" +  
                  str(V) + "\t" + 
                  "\"" + non_lin + "\"");
        if (layerType == 'InnerProduct'):
            non_lin = ""        
            N = params[l][0].data.shape[0]
            C = params[l][0].data.shape[1]
            for i in range (2):
                if (cnn.layers[layerId+i].type == 'ReLU'):
                    non_lin = non_lin + " " + "ReLU"
                if (cnn.layers[layerId+i].type == 'Softmax'):
                    non_lin = non_lin + " " + "Softmax"                    
            print(l      + "\t" + 
                  str(C) + "\t" +
                  str(N) + "\t" + "\t" + "\t" + "\t" + "\t" + 
                  "\"" + non_lin + "\"");

if __name__ == '__main__':
    if (len(sys.argv) == 3):
        protoFile = sys.argv[1]
        modelFile = sys.argv[2]
        ## Display
        print("------------------------------------------------------------------------")
        print("Model: "+ modelFile)
        Topology(protoFile, modelFile)
    else:
        print("Not enought arguments")
        print("python Workload.py <path_to_proto> <path_to_caffemodel>")