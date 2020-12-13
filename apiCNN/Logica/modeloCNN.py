#!/usr/bin/env python
# coding: utf-8

import numpy as np 
import pandas as pd 
import os
from imageio import imread
import PIL
from skimage.transform import resize
from skimage import color
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Convolution2D,BatchNormalization,Flatten,Dense,Dropout,MaxPool2D
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
from keras.models import model_from_json


def cargarRNN(nombreArchivoModelo,nombreArchivoPesos):
    with open(nombreArchivoModelo+'.json', 'r') as f:
        model = model_from_json(f.read())
    model.load_weights(nombreArchivoPesos+'.h5')

    return model


def predict(url):
    CATEGORIES = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
    nombreArchivoModelo='apiCNN/Logica/arquitecturaOptimizados'
    nombreArchivoPesos='apiCNN/Logica/pesosOptimizados'
    model = cargarRNN(nombreArchivoModelo,nombreArchivoPesos)
    #data = imread('/apiCNN/Datasets/flowers/daisy/5547758_eea9edfd54_n.jpg')
    #data = imread('apiCNN/Datasets/flowers/tulip/450607536_4fd9f5d17c_m.jpg')
    #data = imread('apiCNN/Datasets/flowers/dandelion/3730618647_5725c692c3_m.jpg')
    #data = imread('apiCNN/Datasets/flowers/rose/5335944839_a3b6168534_n.jpg')
    #data = imread('/apiCNN/Datasets/flowers/sunflower/8202034834_ee0ee91e04_n.jpg')

    data = imread(url)

    plt.imshow(data,cmap='gray')

    data = resize(data,(100,100))
    data = np.array(data)
    data = data.reshape(data.shape[0], data.shape[1], data.shape[2], 1)
    np.shape(data)

    testImg = data
    testImg=testImg.reshape(1,100,100,3)
    #print('Predicciones:')
    resultados = model.predict(testImg)[0]
    maxElement = np.amax(resultados)
    #print('certeza: ', str(round(maxElement*100, 4))+'%')
    result = np.where(resultados == np.amax(resultados))
    #print('Max :', maxElement)
    #print('Returned tuple of arrays :', result)
    #print('Lista de indices de máximo elemento :', result[0][0])
    index_sample_label=result[0][0]
    #print('Etiqueta predicción: ', CATEGORIES[index_sample_label])
    #print(resultados)
    #plt.show()
    prediccion = dict()
    prediccion['pred']=CATEGORIES[index_sample_label]
    prediccion['prob']=str(round(maxElement, 2))
    return prediccion
