#!/usr/bin/env python

import torch
import torchvision

import base64
import cupy
import cv2
import flask
import getopt
import gevent
import gevent.pywsgi
import h5py
import io
import math
import moviepy
import moviepy.editor
import numpy
import os
import random
import re
import scipy
import scipy.io
import shutil
import sys
import tempfile
import time
import urllib
import zipfile

##########################################################

assert(int(str('').join(torch.__version__.split('.')[0:3])) >= 120) # requires at least pytorch version 1.2.0

torch.set_grad_enabled(False) # make sure to not compute gradients for computational performance

torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance

##########################################################

objectCommon = {}

exec(open('./common.py', 'r').read())

exec(open('./models/disparity-estimation.py', 'r').read())
exec(open('./models/disparity-adjustment.py', 'r').read())
exec(open('./models/disparity-refinement.py', 'r').read())
exec(open('./models/pointcloud-inpainting.py', 'r').read())

##########################################################

arguments_strIn = './images/doublestrike.jpg'
arguments_strOut = './autozoom.mp4'
arguments_centerU = 0.5
arguments_centerV = 0.5
arguments_dblShift = 100
arguments_dblZoom = 1.25
arguments_floatEasyZoom = 1
arguments_floatEasyTurbulence = 0

for strOption, strArgument in getopt.getopt(sys.argv[1:], '', [ strParameter[2:] + '=' for strParameter in sys.argv[1::2] ])[0]:
	if strOption == '--in' and strArgument != '': arguments_strIn = strArgument # path to the input image
	if strOption == '--out' and strArgument != '': arguments_strOut = strArgument # path to where the output should be stored
	if strOption == '--centerU' and strArgument != '': arguments_centerU = float(strArgument) # Start position in the U axis for the zoom
	if strOption == '--centerV' and strArgument != '': arguments_centerV = float(strArgument) # Start position in the V axis for the zoom
	if strOption == '--shift' and strArgument != '': arguments_dblShift = float(strArgument) # Shift value for the zoom
	if strOption == '--zoom' and strArgument != '': arguments_dblZoom = float(strArgument) # Zoom force to be applied
	if strOption == '--easy' and strArgument != '': arguments_floatEasyZoom = float(strArgument) # Easy motion for the zoom in and out
	if strOption == '--turbulence' and strArgument != '': arguments_floatEasyTurbulence = float(strArgument) # Add some turbulence to the movement
# end

##########################################################

if __name__ == '__main__':
	numpyImage = cv2.imread(filename=arguments_strIn, flags=cv2.IMREAD_COLOR)

	intWidth = numpyImage.shape[1]
	intHeight = numpyImage.shape[0]

	dblRatio = float(intWidth) / float(intHeight)

	intWidth = min(int(1024 * dblRatio), 1024)
	intHeight = min(int(1024 / dblRatio), 1024)

	numpyImage = cv2.resize(src=numpyImage, dsize=(intWidth, intHeight), fx=0.0, fy=0.0, interpolation=cv2.INTER_AREA)

	process_load(numpyImage, {})

	objectFrom = {
		'dblCenterU': intWidth * arguments_centerU,
		'dblCenterV': intHeight * arguments_centerV,
		'intCropWidth': int(math.floor(0.97 * intWidth)),
		'intCropHeight': int(math.floor(0.97 * intHeight))
	}

	objectTo = process_autozoom({
		'dblShift': arguments_dblShift,
		'dblZoom': arguments_dblZoom,
		'objectFrom': objectFrom
	})

	numpyResult = process_kenburns({
		'dblSteps': numpy.linspace(0.0, 1.0, 75).tolist(),
		'objectFrom': objectFrom,
		'objectTo': objectTo,
		'easyZoom': arguments_floatEasyZoom, 
		'easyTurbulence': arguments_floatEasyTurbulence,
		'boolInpaint': True
	})

	moviepy.editor.ImageSequenceClip(sequence=[ numpyFrame[:, :, ::-1] for numpyFrame in numpyResult + list(reversed(numpyResult))[1:] ], fps=25).write_videofile(arguments_strOut)
# end