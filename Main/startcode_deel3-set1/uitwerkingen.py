from functools import total_ordering

import numpy as np
from tensorflow import keras
import tensorflow as tf


def load_model():
    # Deze methode laadt het getrainde model dat je bij de vorige opgavenset heb
    # opgeslagen.
    return keras.models.load_model('../startcode_deel2-set3/mnist.keras')

    # YOUR CODE HERE
    pass

# OPGAVE 1a
def conf_matrix(labels, pred):
    # Retourneer de econfusion matrix op basis van de gegeven voorspelling (pred) en de actuele
    # waarden (labels). Check de documentatie van tf.math.confusion_matrix:
    # https://www.tensorflow.org/api_docs/python/tf/math/confusion_matrix
    return tf.math.confusion_matrix(labels, pred)
    pass
    

# OPGAVE 1b
def conf_els(conf, labels):
    # Deze methode krijgt een confusion matrix mee (conf) en een set van labels. Als het goed is, is 
    # de dimensionaliteit van de matrix gelijk aan len(labels) × len(labels) (waarom?). Bereken de 
    # waarden van de TP, FP, FN en TN conform de berekening in de opgave. Maak vervolgens gebruik van
    # de methodes zip() en list() om een list van len(labels) te retourneren, waarbij elke tupel 
    # als volgt is gedefinieerd:

    #     (categorie:string, tp:int, fp:int, fn:int, tn:int)

    # Check de documentatie van numpy diagonal om de eerste waarde te bepalen.
    # https://numpy.org/doc/stable/reference/generated/numpy.diagonal.html

    metrics = list()

    tp = conf.diagonal()
    for i in range(len(labels)):
        fp = sum(conf[:, i]) - tp[i]
        fn = sum(conf[i, :]) - tp[i]
        tn = np.sum(conf) - tp[i] - fp - fn
        metrics.append((labels[i], tp[i], fp, fn, tn))

    return metrics
    pass

# OPGAVE 1c
def conf_data(metrics):
    # Deze methode krijgt de lijst mee die je in de vorige opgave hebt gemaakt (dus met lengte len(labels))
    # Maak gebruik van een list-comprehension om de totale tp, fp, fn, en tn te berekenen en 
    # bepaal vervolgens de metrieken die in de opgave genoemd zijn. Retourneer deze waarden in de
    # vorm van een dictionary (de scaffold hiervan is gegeven).

    # VERVANG ONDERSTAANDE REGELS MET JE EIGEN CODE
    tp, fp, fn, tn = 0, 0, 0, 0

    for i in range(len(metrics)):
        tp += metrics[i][1]
        fp += metrics[i][2]
        fn += metrics[i][3]
        tn += metrics[i][4]

    tpr = tp / (tp + fn)
    ppv = tp / (tp + fp)
    tnr = tn / (tn + fp)
    fpr = fp / (fp + tn)

    rv = {'tpr':tpr, 'ppv':ppv, 'tnr':tnr, 'fpr':fpr }
    return rv