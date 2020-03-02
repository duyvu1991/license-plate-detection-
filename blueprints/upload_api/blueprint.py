from flask import Flask, Blueprint, render_template, request
import tensorflow as tf
import re
import base64
import numpy as np
import pathlib as p
from .model_utils import load_model, detect_lp, im2single, predict_number
import cv2
import os

#load model to detect license plate
wpod_net_path = "models/wpod-net_update1.json"
wpod_net = load_model(wpod_net_path)

#load model to recognize the number on plate
model_svm = cv2.ml.SVM_load('models/svm.xml')

    
upload_api = Blueprint('upload_api', __name__)
@upload_api.route('/upload', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
        f = request.files['image']
        f.save(os.path.join(p.Path.cwd() , 'static' , 'images' , str(f.filename)))

        # Read file
        img_path = os.path.join(p.Path.cwd() , 'static' , 'images' , str(f.filename))
        Ivehicle = cv2.imread(img_path)


        # Define size 
        Dmax = 608
        Dmin = 288

        # Ratio W / H and min(dim)
        ratio = float(max(Ivehicle.shape[:2])) / min(Ivehicle.shape[:2])
        side = int(ratio * Dmin)
        bound_dim = min(side, Dmax)

        _ , LpImg, lp_type = detect_lp(wpod_net, im2single(Ivehicle), bound_dim, lp_threshold=0.5)


        if (len(LpImg)):
            bs = cv2.convertScaleAbs(LpImg[0], alpha=(255.0))
            cv2.imwrite(os.path.join(p.Path.cwd() , 'static' , 'images' , "final_" + str(f.filename)),bs)

        label = predict_number(os.path.join(p.Path.cwd() , 'static' , 'images' , "final_" + str(f.filename)), model_svm)

        return render_template('predict.html',label = label , img_name= ("final_" + f.filename))