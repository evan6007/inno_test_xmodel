#!/usr/bin/python3
# Copyright (c) 2022 Innodisk Crop.
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT


import os
import argparse
import time
import numpy as np
import queue
import logging
import datetime
import sys

from mod.predictor import PREDICTOR
from mod.util import open_json

DISPLAY_CARD_PATH = '/dev/dri/by-path/platform-fd4a0000.zynqmp-display-card'

version = "1.0.0"
date = datetime.date.today()

CFG     = 'config.json'
DISPLAY = "DISPLAY"
WIDTH   = "WIDTH"
HEIGHT  = "HEIGHT"
divider = '------------------------------------'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-c', '--camera', type=str, help='Camera type, default is /dev/video0')
    ap.add_argument('-i', '--image' , type=str, default='Sign-Language-Digits-Dataset/Dataset/0/IMG_1118.JPG', help='Path to a image which you would like to inference.')
    ap.add_argument('-t', '--target', type=str, default='image', help='Output type, default is dp')
    ap.add_argument('-v', '--video' , type=str, help='Path to a video which you would like to inference.')
    ap.add_argument('-x', '--xmodel', type=str, default='handsign', help='Type of xmodel, default is yolo')
    ap.add_argument('-l', '--lpr'   , type=str, help='Enable LPR mode. This is base on yolo object detection.')
    
    args = ap.parse_args()
    cfg = open_json(CFG)
    pred = PREDICTOR(args, cfg)

    logging.basicConfig(level=logging.INFO)

    logging.info(divider)
    logging.info(" Command line options:")
    logging.info(" Version: {}".format(version))
    logging.info(" Date:    {}".format(date))

    ''' Select Input '''

    if args.image:
        if os.path.isfile(args.image):
            logging.info(' --input     : {}'.format('image'))
            pred.get_frame = pred.image_get

        else:
            print("Could not find the file {}.".format(args.image))
            return


    ''' Select Xmodel '''
    if args.xmodel == 'cnn':
        logging.info(' --model     : {}'.format('cnn'))
        pred.init_model = pred.init_cnn
        pred.run_model = pred.run_cnn

    ''' Select Output '''

    if args.target == 'image':
        logging.info(' --output    : {}'.format('image'))
        pred.output = pred.image_out



    pred.predict()

if __name__ == '__main__':
    main()