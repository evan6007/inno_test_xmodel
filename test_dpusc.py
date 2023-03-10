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
    ap.add_argument('-i', '--image' , type=str, help='Path to a image which you would like to inference.')
    ap.add_argument('-t', '--target', type=str, help='Output type, default is dp')
    ap.add_argument('-v', '--video' , type=str, help='Path to a video which you would like to inference.')
    ap.add_argument('-x', '--xmodel', type=str, default='yolo', help='Type of xmodel, default is yolo')
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
    # if args.camera == 'g4':
    #     logging.info(' --input     : {}'.format('g4'))
    #     pred.get_frame = pred.g4cam_get

    if args.camera:
        if (args.camera.isdigit()):
            logging.info(' --input     : {}'.format('web-cam'))
            pred.get_frame = pred.cam_get

        else:
            print("string, please input digital number")
            return

    elif args.image:
        if os.path.isfile(args.image):
            logging.info(' --input     : {}'.format('image'))
            pred.get_frame = pred.image_get

        else:
            print("Could not find the file {}.".format(args.image))
            return

    elif args.video:
        if os.path.isfile(args.video):
            logging.info(' --input     : {}'.format('video'))
            pred.get_frame = pred.video_get

        else:
            print("Could not find the file {}.".format(args.image))
            return



    ''' Select Xmodel '''
    if args.xmodel == 'cnn':
        logging.info(' --model     : {}'.format('cnn'))
        pred.init_model = pred.init_cnn
        pred.run_model = pred.run_cnn

    else:
        logging.info(' --model     : {}'.format('yolo'))
        pred.init_model = pred.init_yolo
    
        if args.lpr:
            logging.info(' --lpr     : {}'.format('enable'))
            pred.run_model = pred.run_yolo_lpr
        else:
            pred.run_model = pred.run_yolo

    ''' Select Output '''
    if args.target == 'dp':
        if not os.path.exists(DISPLAY_CARD_PATH):
            logging.info('Error: zynqmp-display device is not ready.')
            return

        ''' Resolution check '''
        width = int(pred.cfg[DISPLAY][WIDTH])
        height = int(pred.cfg[DISPLAY][HEIGHT])
        
        resolution = "{}x{}".format(width, height)
        all_res = os.popen("modetest -M xlnx -c| awk '/name refresh/ {f=1;next}  /props:/{f=0;} f{print $1 \"@\" $2}'").read()
        
        if all_res.find(resolution) == -1:
            os.environ["DISPLAY"] = ":0"

        logging.info(' --output    : {}'.format('dp'))
        pred.output = pred.dp_out

    elif args.target == 'image':
        logging.info(' --output    : {}'.format('image'))
        pred.output = pred.image_out

    elif args.target == 'video':
        logging.info(' --output    : {}'.format('video'))
        pred.output = pred.video_out


    pred.predict()

if __name__ == '__main__':
    main()