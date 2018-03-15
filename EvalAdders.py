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

CAFFE_ROOT       = os.environ['CAFFE_ROOT']
CAFFE_PYTHON_LIB = CAFFE_ROOT+'/python'
sys.path.insert(0, CAFFE_PYTHON_LIB)
os.environ['GLOG_minloglevel'] = '2' # Supresses Display on console
import caffe;

def QuantizeParam(real_param,bit_width=8):
	scale_factor = math.pow(2,(bit_width - 1)) - 1
	quantized_param = scale_factor * real_param
	return np.array(np.round(quantized_param),dtype=int)

def EvalAdders(protoFile, modelFile):
	num_add_tot = np.array([],dtype=int)
	cnn = caffe.Net(protoFile,1,weights=modelFile)
	params = cnn.params
	blobs = cnn.blobs
	# Print CNN shape
	for l in cnn._layer_names:
		layer_id = list(cnn._layer_names).index(l)
		layer_type =  cnn.layers[layer_id].type
		if (layer_type == 'Convolution'):
			num_add_layer = np.array([],dtype=int)
			q_param =  QuantizeParam(params[l][0].data)
			N = q_param.shape[0]
			C = q_param.shape[1]
			J = q_param.shape[2]
			K = q_param.shape[3]
			for n in range(N):
				num_add_layer = np.append(num_add_layer,
				                          np.count_nonzero(q_param[n]))
			num_add_tot = np.append(num_add_tot,
			                        np.array(np.mean(num_add_layer),dtype=int))
	print(num_add_tot)
	return num_add_tot
if __name__ == '__main__':
	if (len(sys.argv) == 3):
		protoFile = sys.argv[1]
		modelFile = sys.argv[2]
		## Display
		print("------------------------------------------------------------------------")
		print("Model: "+ modelFile)
		num_adders = EvalAdders(protoFile, modelFile)
