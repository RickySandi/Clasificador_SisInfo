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
palabras_parada = set( nltk.corpus.stopwords.words('english') + list(string.punctuation)+["...","..","hr"])

def cleanHtml(html):
    limpiar_html = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
    clean_html = re.sub(limpiar_html, '', html)
    return clean_html

def delete_html(palabras):
    array = []
    for palabra in  palabras:
        deleted_html = cleanHtml(palabra)
        array.append(deleted_html)
    return array

def delete_quotes(palabras):
    array = []
    for palabra in  palabras:
        palabra = palabra.strip("'")
        palabra = palabra.strip("`")
        array.append(palabra)
    return array

def tokenize(texto):
    texto=texto.lower()
    palabras = nltk.word_tokenize(texto)#separa las palabras
    return [palabra for palabra in palabras if palabra not in palabras_parada]#quita stopwords
def list_to_string(lista):  
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
    tokenizado=tokenize(data['Review'])
    cleanHtml=delete_html(tokenizado)
    sinComillas=delete_quotes(cleanHtml)
    comentario=list_to_string(sinComillas)  
    one_hot_vec = one_hot_vector(comentario, vocabulary)
    predict_request = np.array(list(one_hot_vec)).reshape(1,-1)
    y_hat = classifier.predict(predict_request)

    if int(y_hat[0]) is 1:
        y_hat = "pos"
    else:
        y_hat = "neg"
    output = {'Review':data['Review'], 'label': y_hat} 
    return output
    

if __name__ == '__main__':
    app.run(port=5003, debug=True)



