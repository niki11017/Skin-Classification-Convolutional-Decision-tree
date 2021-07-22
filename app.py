from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

import flask 
import pickle
import pandas as pd
import pickle
import joblib
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template,session
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

app = flask.Flask(__name__, template_folder='templates') 

app = flask.Flask(__name__, static_url_path='/static')

app.static_folder = 'static'
# Model saved with Keras model.save()
MODEL_PATH = 'models/model_feature.h5'

# Load your trained model
model = load_model(MODEL_PATH)
#model._make_predict_function()          # Necessary
# print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
#model.save('')
print('Model loaded. Check http://127.0.0.1:5000/')

model_final=joblib.load('models/modelfinal.pkl')
print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, model):
   
    img = image.load_img(img_path, target_size=(71,71))

    # Preprocessing the image
    x = image.img_to_array(img)
    #x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)


    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x = preprocess_input(x, mode='caffe')

    preds = model.predict(x)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')





@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)

        # Process your result for human
        # pred_class = preds.argmax(axis=-1) 
        from flask import jsonify
        array=preds.tolist()
        session['array']=array

        return redirect(url_for('.test', array=array))
    

@app.route('/test', methods=['GET', 'POST'])
def test():
    if flask.request.method == 'GET':
        return(flask.render_template('test.html'))
    if flask.request.method == 'POST':
        female = flask.request.form['female']
        male = flask.request.form['male']
        abdomen = flask.request.form['abdomen']
        acral= flask.request.form['acral']
        back= flask.request.form['back']
        chest= flask.request.form['chest']
        ear= flask.request.form['ear']
        face= flask.request.form['face']
        foot= flask.request.form['foot']
        genital= flask.request.form['genital']
        hand= flask.request.form['hand']
        lower_extremity= flask.request.form['lower_extremity']
        neck=flask.request.form['neck']
        scalp= flask.request.form['scalp']
        trunk= flask.request.form['trunk']
        upper_extremity= flask.request.form['upper_extremity']
        unknown= flask.request.form['unknown']
        confocal= flask.request.form['confocal']
        consensus=flask.request.form['consensus']
        follow_up= flask.request.form['follow_up']
        histo= flask.request.form['histo']
        age= flask.request.form['age']
        print("values taken in")


        # Make prediction
        #nos=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51]
        #no=['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35','36','37','38','39','40','41','42','43','44','45','46','47','48','49','50','51']
        #dicto=dict(zip(nos,*array))
        
        #input_variables = pd.DataFrame([[female, male, abdomen, acral, back, chest, ear, face,foot, genital, hand, lower_extremity, neck, scalp, trunk,unknown, upper_extremity, confocal, consensus, follow_up,histo, age,array[0][0],  array[0][1], array[0][2],  array[0][3],  array[0][4],  array[0][5],  array[0][6],  array[0][7],  array[0][8],  array[0][9],  array[0][10],array[0][11],  array[0][12], array[0][13],  array[0][14],  array[0][15],  array[0][16],  array[0][17],  array[0][18], array[0][19],  array[0][20],  array[0][21],  array[0][22],array[0][23],  array[0][24], array[0][25],  array[0][26],  array[0][27],  array[0][28],  array[0][29],  array[0][30],  array[0][31],  array[0][32],  array[0][33],  array[0][34],array[0][35],  array[0][36],  array[0][37],  array[0][38],  array[0][39],  array[0][40],  array[0][41],  array[0][42],  array[0][43],  array[0][44],  array[0][45],  array[0][46],array[0][47],  array[0][48],  array[0][49],  array[0][50],  array[0][51] ]],columns=['female', 'male', 'abdomen', 'acral', 'back', 'chest', 'ear', 'face','foot', 'genital', 'hand', 'lower extremity', 'neck', 'scalp', 'trunk','unknown', 'upper extremity', 'confocal', 'consensus', 'follow_up','histo', 'age', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15' , '16', '17', '18', '19', '20', '21', '22','23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34','35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46','47', '48', '49', '50', '51'],dtype=float)
        array=session.get('array',None)
        input_variables = pd.DataFrame([[female, male, abdomen, acral, back, chest, ear, face,
                                          foot, genital, hand, lower_extremity, neck, scalp, trunk,
                                              unknown, upper_extremity, confocal, consensus, follow_up,
                                                 histo, age, array[0][0],  array[0][1], array[0][2],  array[0][3],  array[0][4],  array[0][5],  array[0][6],  array[0][7],  array[0][8],  array[0][9],  array[0][10],
                                                        array[0][11],  array[0][12], array[0][13],  array[0][14],  array[0][15],  array[0][16],  array[0][17],  array[0][18], array[0][19],  array[0][20],  array[0][21],  array[0][22],
                                                     array[0][23],  array[0][24], array[0][25],  array[0][26],  array[0][27],  array[0][28],  array[0][29],  array[0][30],  array[0][31],  array[0][32],  array[0][33],  array[0][34],
                                                      array[0][35],  array[0][36],  array[0][37],  array[0][38],  array[0][39],  array[0][40],  array[0][41],  array[0][42],  array[0][43],  array[0][44],  array[0][45],  array[0][46],
                                                      array[0][47],  array[0][48],  array[0][49],  array[0][50],  array[0][51] ]],
                                       columns=['female', 'male', 'abdomen', 'acral', 'back', 'chest', 'ear', 'face',
                                                'foot', 'genital', 'hand', 'lower extremity', 'neck', 'scalp', 'trunk',
                                                'unknown', 'upper extremity', 'confocal', 'consensus', 'follow_up',
                                                'histo', 'age', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
                                                '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22',
                                                 '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34',
                                                '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46',
                                                 '47', '48', '49', '50', '51'],
                                       dtype=float)      
        print(input_variables)
        prediction = model_final.predict(input_variables)[0]
        print(prediction)

        return flask.render_template('test.html',
        original_input={'female':female, 
                                             'male':male,
                                             'abdomen':abdomen,
                                             'acral':acral,
                                             'back':back,
                                             'chest':chest,
                                             'ear':ear,
                                             'face':face,
                                             'foot':foot,
                                             'genital':genital,
                                             'hand':hand, 
                                             'lower extremity':lower_extremity,
                                                           'neck':neck,
                                                            'scalp':scalp,
                                                             'trunk':trunk,
                                                             'unknown':unknown, 
                                                             'upper extremity':upper_extremity,
                                                              'confocal':confocal,
                                                               'consensus':consensus,
                                                                'follow_up':follow_up,
                                                                 'histo':histo, 
                                                                 'age':age, 
                                                                 '0':array[0][0], 
                                                                 '1':array[0][1],
                                                                  '2':array[0][2],
                                                                   '3':array[0][3],
                                                                    '4':array[0][4],
                                                                     '5':array[0][5],
                                                                      '6':array[0][6],
                                                                       '7':array[0][7],
                                                                        '8':array[0][8],
                                                                         '9':array[0][9],
                                                                          '10':array[0][10],
                                                                          '11':array[0][11],
                                                                           '12':array[0][12],
                                                                            '13':array[0][13],
                                                                             '14':array[0][14],
                                                                              '15':array[0][15],
                                                                               '16':array[0][16],
                                                                                '17':array[0][17],
                                                                                 '18':array[0][18],
                                                                                  '19':array[0][19],
                                                                                   '20':array[0][20],
                                                                                    '21':array[0][21],
                                                                                     '22':array[0][22],
                                                                              '23':array[0][23], '24':array[0][24], '25':array[0][25], '26':array[0][26], '27':array[0][27],
                                                                               '28':array[0][28], '29':array[0][29], '30':array[0][30], '31':array[0][31], '32':array[0][32], '33':array[0][33],
                                                                                '34':array[0][34],
                                                                               '35':array[0][35], '36':array[0][36], '37':array[0][37], '38':array[0][38], '39':array[0][39], 
                                                                               '40':array[0][40], '41':array[0][41], '42':array[0][42], '43':array[0][43], '44':array[0][44], '45':array[0][45],
                                                                                '46':array[0][46],
                                                                                '47':array[0][47], '48':array[0][48], '49':array[0][49], '50':array[0][50], '51':array[0][51]
                                                     },
                                     
                                                     
                                                    
                                     result=prediction,
                                     )

        
   

if __name__ == '__main__':
    app.secret_key = 'super secret key'
    app.config['SESSION_TYPE'] = 'filesystem'
    app.debug = True
    app.run()

