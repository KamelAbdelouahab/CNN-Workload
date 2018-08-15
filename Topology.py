#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys
import os
import io
import numpy as np
import math
import matplotlib.pyplot as plt
import pylab as P

try:
    CAFFE_ROOT = os.environ['CAFFE_ROOT']
    CAFFE_PYTHON_LIB = CAFFE_ROOT+'/python'
    sys.path.insert(0, CAFFE_PYTHON_LIB)
except KeyError:
    print("Warning: CAFFE_ROOT environment variable not set")
os.environ['GLOG_minloglevel'] = '2' # Supresses Display on console
import caffe;

def FormatedPrint(layer_name,W,H,C,V,U,N,K,s,p):
    if "layer" in layer_name:
        x = layer_name + "\t"
    else:
        x = layer_name + "\t\t"
    print (x +
          str(W) + "\t" +
          str(H) + "\t" +
          str(C) + "\t" +
          str(V) + "\t" +
          str(U) + "\t" +
          str(N) + "\t" +
          str(K) + "\t" +
          str(s) + "\t" +
          str(p) + ""
          );

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

            for i in range (1,6):
                try:
                    cnn.layers[layerId+i]
                    if (cnn.layers[layerId+i].type == 'Convolution'):
                        break
                    if (cnn.layers[layerId+i].type == 'ReLU'):
                        non_lin = non_lin + "+" + "ReLU"
                    if (cnn.layers[layerId+i].type == 'Scale'):
                        non_lin = non_lin + "+" + "Scale"
                    if (cnn.layers[layerId+i].type == 'TanH'):
                        non_lin = non_lin + "+" + "TanH"
                    if (cnn.layers[layerId+i].type == 'Sigmoid'):
                        non_lin = non_lin + "+" + "Sigmoid"
                    if (cnn.layers[layerId+i].type == 'Pooling'):
                        non_lin = non_lin + "+" + "Pool"
                    if (cnn.layers[layerId+i].type == 'LRN' or cnn.layers[layerId+i].type == "BatchNorm"):
                        non_lin = non_lin + "+" + "BN"
                except IndexError:
                    pass

            print((l      + "\t" +
                  str(C) + "\t" +
                  str(N) + "\t" +
                  str(J) + "x" +
                  str(K) + "\t" +
                  str(U) + "x" +
                  str(V) + "\t" +
                  "" + non_lin + ""));
        if (layerType == 'InnerProduct'):
            non_lin = ""
            N = params[l][0].data.shape[0]
            C = params[l][0].data.shape[1]
            for i in range (4):
                try:
                    cnn.layers[layerId+i]
                    if (cnn.layers[layerId+i].type == 'TanH'):
                        non_lin = non_lin + "+" + "TanH"
                    if (cnn.layers[layerId+i].type == 'ReLU'):
                        non_lin = non_lin + "+" + "ReLU"
                    if (cnn.layers[layerId+i].type == 'Softmax'):
                        non_lin = non_lin + "+" + "Softmax"
                except IndexError:
                    pass
            print((l      + "\t" +
                  str(C) + "\t" +
                  str(N) + "\t" + "\t" +  "\t" +
                  "" + non_lin + ""));

def CompleteTopology(protoFile, modelFile):
    from caffe.proto import caffe_pb2
    from google.protobuf import text_format
    parsible_net = caffe_pb2.NetParameter()
    text_format.Merge(open(protoFile).read(), parsible_net)
    cnn     = caffe.Net(protoFile,1,weights=modelFile)
    params  = cnn.params
    blobs   = cnn.blobs
    print("----------------------------------------------------------------------------------")
    print("layer\t\tW\tH\tC\tV\tU\tN\tK\ts\tp")
    print("----------------------------------------------------------------------------------")

    for layer in parsible_net.layer:
        if (layer.type == 'Input'):
            l = layer.name
            H = blobs[l].data.shape[2]
            W = blobs[l].data.shape[3]
        if (layer.type == 'Convolution'):
            l = layer.name
            N = layer.convolution_param.num_output if len(layer.convolution_param.kernel_size) else 1
            C = params[l][0].data.shape[1]
            K = layer.convolution_param.kernel_size[0] if len(layer.convolution_param.kernel_size) else 1
            p = layer.convolution_param.pad[0] if len(layer.convolution_param.pad) else 0
            s = layer.convolution_param.stride[0] if len(layer.convolution_param.stride) else 1
            U = blobs[l].data.shape[2]
            V = blobs[l].data.shape[3]
            FormatedPrint(l,W,H,C,V,U,N,K,s,p)
            H = blobs[l].data.shape[2]
            W = blobs[l].data.shape[3]

        if (layer.type == 'Pooling'):
            l = layer.name
            N = blobs[l].data.shape[1]
            C = blobs[l].data.shape[1]
            K = layer.pooling_param.kernel_size
            p = layer.pooling_param.pad
            s = layer.pooling_param.stride
            U = blobs[l].data.shape[2]
            V = blobs[l].data.shape[3]
            FormatedPrint(l,W,H,C,V,U,N,K,s,p)
            H = blobs[l].data.shape[2]
            W = blobs[l].data.shape[3]
            print("----------------------------------------------------------------------------------")
        if (layer.type == "ReLU"):
            print(layer.name)

        if (layer.type == "LRN"):
            print(layer.name)

        if (layer.type == "BatchNorm"):
            print(layer.name)

        if (layer.type == "Scale"):
            print(layer.name)

if __name__ == '__main__':
    if (len(sys.argv) == 3):
        protoFile = sys.argv[1]
        modelFile = sys.argv[2]
        ## Display
        print("----------------------------------------------------------------------------------")
        print(("Model: "+ modelFile))
        CompleteTopology(protoFile, modelFile)
        print("----------------------------------------------------------------------------------")
    else:
        print("Not enought arguments")
        print("python Workload.py <path_to_proto> <path_to_caffemodel>")
