import pandas as pd
import numpy as np
import nltk
import string
from nltk.tokenize import sent_tokenize
import os
from flask import Flask, request, render_template, jsonify, make_response
import pickle
import joblib
import re

classifier_filepath = os.path.join("tree_v4.pkl")
classifier_file = open(classifier_filepath, "rb")
classifier = pickle.load(open(classifier_filepath, "rb"))
classifier_file.close()

vocabulary_filepath = os.path.join("VocabularioProblema.pkl")
vocabulary_file = open(vocabulary_filepath, "rb")
vocabulary = pickle.load(open(vocabulary_filepath, "rb"))
vocabulary_file.close()

app = Flask(__name__)
palabrasParada = set( nltk.corpus.stopwords.words('english') + list(string.punctuation)+["...","..","hr"])

def sinhtml(conHTML):
    limpiar_html = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
    sin_html = re.sub(limpiar_html, '', conHTML)
    return sin_html

def quitar_html(palabras):
    array = []
    for palabra in  palabras:
        sinHtml = sinhtml(palabra)
        array.append(sinHtml)
    return array

def quitarComillas(palabras):
    array = []
    for palabra in  palabras:
        palabra = palabra.strip("'")
        palabra = palabra.strip("`")
        array.append(palabra)
    return array

def Tokenizar(texto):
    texto=texto.lower()
    palabras = nltk.word_tokenize(texto)#separa las palabras
    return [palabra for palabra in palabras if palabra not in palabrasParada]#quita stopwords
def listToString(lista):  
    str1 = " " 
    return (str1.join(lista)) 

def one_hot_vector(document, problem_vocabulary):
    vector = np.zeros(len(problem_vocabulary),dtype=int)
    for token in document.split():
        if token in problem_vocabulary:
            vector[problem_vocabulary[token]] = 1
    return vector


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    tokenizado=Tokenizar(data['Review'])
    sinHTML=quitar_html(tokenizado)
    sinComillas=quitarComillas(sinHTML)
    comentario=listToString(sinComillas)  
    one_hot_vec = one_hot_vector(comentario, vocabulary)
    predict_request = np.array(list(one_hot_vec)).reshape(1,-1)
    y_hat = classifier.predict(predict_request)
    # y_hat_2 = y_hat
    
    if int(y_hat[0]) == 1:
        y_hat = "pos"
    else:
        y_hat = "neg"
    output = {'Review':data['Review'], 'label': y_hat} 
    return output

@app.route('/', methods=['GET'])
def prueba():
    return "Entra al get"
    

if __name__ == '__main__':
    app.run(port=12345, debug=True)



