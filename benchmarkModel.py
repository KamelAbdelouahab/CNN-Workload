#!/usr/bin/env python
# -*- coding: UTF-8 -*-

##----------------------------------------------------------------------------
## Title      : benchmarkModel
##----------------------------------------------------------------------------
## File       : benchmarkModel.py
## Author     : Kamel Abdelouahab
## Company    : Institut Pascal - DREAM
## Last update: 18-10-2017
##----------------------------------------------------------------------------

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


def listLayers(protoFile, modelFile):
	cnn           = caffe.Net(protoFile,1,weights=modelFile)
	params	      = cnn.params
	blobs         = cnn.blobs
	#~ print("\n".join(map(str, cnn._layer_names)))
	#~ print blobs
	for layerName in cnn._layer_names:
		layerId = list(cnn._layer_names).index(layerName)
		layerType =  cnn.layers[layerId].type
		print layerName,"\t",layerType


def workload(protoFile, modelFile):
	convLayerName = np.array([])
	convWorkload  = np.array([])
	convMemory	  = np.array([])

	fcLayerName	  = np.array([])	
	fcWorkload 	  = np.array([])
	fcMemory 	  = np.array([])
	
	numPool = 0
	
	cnn           = caffe.Net(protoFile,1,weights=modelFile)
	params	      = cnn.params
	blobs         = cnn.blobs
	# Print CNN shape	
	for l in cnn._layer_names:
		layerId = list(cnn._layer_names).index(l)
		layerType =  cnn.layers[layerId].type
		
		if (layerType == 'Convolution'):
			convLayerName = np.append(convLayerName,l)		# Layer Name			
			N = params[l][0].data.shape[0]
			C = params[l][0].data.shape[1]
			K = params[l][0].data.shape[2]
			U = blobs[l].data.shape[2]
			nonZeros = np.count_nonzero(params[l][0].data)
			# Convolution Workload
			thisLayerWorkload = U*U*nonZeros #~ thisLayerWorkload = U*U*K*K*N*C
			convWorkload = np.append(convWorkload,thisLayerWorkload)
			# Param size
			thisLayerMemory = nonZeros
			convMemory = np.append(convMemory,thisLayerMemory)
			
		if (layerType == 'InnerProduct'):
			fcLayerName = np.append(fcLayerName,l)			
			N = params[l][0].data.shape[0]
			C = params[l][0].data.shape[1]
			nonZeros = np.count_nonzero(params[l][0].data)
			thisLayerWorkload = nonZeros #thisLayerWorkload = N*C
			fcWorkload = np.append(fcWorkload,thisLayerWorkload)
		
		if (layerType == 'Pooling'):
			numPool = numPool + 1
			
	fcMemory = fcWorkload
	return convWorkload,fcWorkload, convMemory, fcMemory, numPool

# Found it in Stackoverflow, thanks for the time dude ...
def human_format(num):
    num = float('{:.3g}'.format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'K', 'M', 'B', 'T'][magnitude])
    

if __name__ == '__main__':
    if (len(sys.argv) == 3):
		protoFile = sys.argv[1]
		modelFile = sys.argv[2]
		#~ listLayers(protoFile,modelFile)
		[convWorkload,fcWorkload, convMemory, fcMemory, numPool] =  workload(protoFile,modelFile)
		
		# Results
		## Computations
		totalConvWorkload = np.sum(convWorkload)
		totalFcWorkload   = np.sum(fcWorkload)
		totalWorkload     = totalConvWorkload + totalFcWorkload
		## Memory
		totalConvMemory   = np.sum(convMemory)
		totalFcMemory 	  = np.sum(fcMemory)
		totalMemory 	  = totalConvMemory + totalFcMemory
		## Number of Layers
		numConv = convWorkload.shape[0]
		numFc 	= fcWorkload.shape[0]
		## Display
		print "------------------------------------------------------------------------"
		print "Model: "+ modelFile
		print "Number of conv layers :", numConv
		print "Computational workload of conv layers: ", human_format(totalConvWorkload), "MACs"
		print "Number of parameters of conv layers: ", human_format(totalConvMemory)
		print "------------------------------------------------------------------------"
		print "Number of pooling layers: ", numPool
		print "------------------------------------------------------------------------"
		print "Number of FC layers: ", numFc
		print "Computational workload of FC layers: ", human_format(totalFcWorkload),"MACs"
		print "Number of parameters of FC layers: ", human_format(totalFcMemory)
		print "------------------------------------------------------------------------"
		print "Total Computational workload: ", human_format(totalWorkload),"MACs"
		print "Total Number of parameters: ", human_format(totalMemory)
		print "------------------------------------------------------------------------"
		#~ #ConvWorkload = totalConvWorkload/totalWorkload
		#~ #ConvMemory   = totalConvMemory/totalMemory
		#~ #print "Convolutions",ConvWorkload,"Computations and ",ConvMemory,"Memory"
		
		
