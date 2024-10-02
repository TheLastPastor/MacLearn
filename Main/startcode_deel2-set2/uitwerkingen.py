
import numpy as np
import matplotlib.pyplot as plt
from astropy.wcs.docstrings import delta
from numpy.ma.core import ravel
from scipy.sparse import csr_matrix

# ==== OPGAVE 1 ====
def plot_number(nrVector):
    # Let op: de manier waarop de data is opgesteld vereist dat je gebruik maakt
    # van de Fortran index-volgorde – de eerste index verandert het snelst, de 
    # laatste index het langzaamst; als je dat niet doet, wordt het plaatje 
    # gespiegeld en geroteerd. Zie de documentatie op 
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.reshape.html
    square_vector = np.reshape(nrVector, (20,20), order='F')
    plt.matshow(square_vector, cmap='gray')
    plt.show()

    pass

# ==== OPGAVE 2a ====
def sigmoid(z):
    # Maak de code die de sigmoid van de input z teruggeeft. Zorg er hierbij
    # voor dat de code zowel werkt wanneer z een getal is als wanneer z een
    # vector is.
    # Maak gebruik van de methode exp() in NumPy.
    return 1 / (1 + np.exp(-z))



# ==== OPGAVE 2b ====
def get_y_matrix(y, m):
    # Gegeven een vector met waarden y_i van 1...x, retourneer een (ijle) matrix
    # van m × x met een 1 op positie y_i en een 0 op de overige posities.
    # Let op: de gegeven vector y is 1-based en de gevraagde matrix is 0-based,
    # dus als y_i=1, dan moet regel i in de matrix [1,0, ... 0] zijn en als
    # y_i=10, dan is regel i in de matrix [0,0, ... 1] (10 symboliseert in deze dataset dus 9 en niet 0!).
    # In dit geval is de breedte van de matrix 10 (0-9),
    # maar de methode moet werken voor elke waarde van y en m

    data = np.ones(m,)
    row = np.arange(m,)
    y = y - 1
    y = y.ravel()
    return csr_matrix((data, (row, y)), shape=(m, 10)).todense()

    pass

# ==== OPGAVE 2c ==== 
# ===== deel 1: =====
def predict_number(Theta2, Theta3, X):
    # Deze methode moet een matrix teruggeven met de output van het netwerk
    # gegeven de waarden van Theta2 en Theta3. Elke regel in deze matrix 
    # is de waarschijnlijkheid dat het sample op die positie (i) het getal
    # is dat met de kolom correspondeert.

    # De matrices Theta2 en Theta3 corresponderen met het gewicht tussen de
    # input-laag en de verborgen laag, en tussen de verborgen laag en de
    # output-laag, respectievelijk. 

    # Een mogelijk stappenplan kan zijn:

    #    1. voeg enen toe aan de gegeven matrix X; dit is de input-matrix a1
    #    2. roep de sigmoid-functie van hierboven aan met a1 als actuele
    #       parameter: dit is de variabele a2
    #    3. voeg enen toe aan de matrix a2, dit is de input voor de laatste
    #       laag in het netwerk
    #    4. roep de sigmoid-functie aan op deze a2; dit is het uiteindelijke
    #       resultaat: de output van het netwerk aan de buitenste laag.

    # Voeg enen toe aan het begin van elke stap en reshape de uiteindelijke
    # vector zodat deze dezelfde dimensionaliteit heeft als y in de exercise.

    a1 = np.c_[np.ones(X.shape[0]), X]
    a2 = sigmoid(np.matmul(a1, Theta2.T))
    a2 = np.c_[np.ones(a2.shape[0]), a2]
    a3 = sigmoid(np.matmul(a2, Theta3.T))

    return a3

    pass



# ===== deel 2: =====
def compute_cost(Theta2, Theta3, X, y):
    # Deze methode maakt gebruik van de methode predictNumber() die je hierboven hebt
    # geïmplementeerd. Hier wordt het voorspelde getal vergeleken met de werkelijk 
    # waarde (die in de parameter y is meegegeven) en wordt de totale kost van deze
    # voorspelling (dus met de huidige waarden van Theta2 en Theta3) berekend en
    # geretourneerd.
    # Let op: de y die hier binnenkomt is de m×1-vector met waarden van 1...10. 
    # Maak gebruik van de methode get_y_matrix() die je in opgave 2a hebt gemaakt
    # om deze om te zetten naar een ijle matrix.
    y_matrix = get_y_matrix(y, y.shape[0])

    predictions = predict_number(Theta2, Theta3, X)
    # log loss cost function
    # cost = -(1/len(y)) * np.sum(y_matrix * np.log(predictions) + (1 - y_matrix) * np.log(1 - predictions))
    cost = -(1 / y.shape[0]) * np.sum(np.multiply(y_matrix, np.log(predictions)) + np.multiply((1 - y_matrix), np.log(1 - predictions)))

    return cost



# ==== OPGAVE 3a ====
def sigmoid_gradient(z): 
    # Retourneer hier de waarde van de afgeleide van de sigmoïdefunctie.
    # Zie de opgave voor de exacte formule. Controleer dat deze werkt met
    # scalaire waarden en met vectoren.
    sigmoid_value = sigmoid(z)
    return sigmoid_value * (1 - sigmoid_value)

# ==== OPGAVE 3b ====
def nn_check_gradients(Theta2, Theta3, X, y):
    # gewichten input laag en verborgen laag
    Delta2 = np.zeros(Theta2.shape)
    # gewichten verborgen laag en output laag
    Delta3 = np.zeros(Theta3.shape)

    m = X.shape[0]
    for i in range(m):
        # add bias aan inputvector
        a1 = np.r_[1, X[i]]
        # forward propagation
        z2 = np.dot(Theta2, a1) # activatie verborgen laag
        a2 = np.r_[1, sigmoid(z2)] # output verborgen laag na sigmoid functie en add bias
        z3 = np.dot(Theta3, a2) # activatie output laag
        a3 = sigmoid(z3) # output laag

        # Bereken error voor de output layer
        y_matrix = get_y_matrix(y, m)
        delta3 = a3 - y_matrix[i, :].A1 # fout in de outputlaag

        # Bereken error voor de hidden layer 
        delta2 = np.dot(Theta3.T, delta3) * sigmoid_gradient(np.r_[1, z2])
        delta2 = delta2[1:]  # Verwijder bias

        # Sum gradients
        Delta3 += np.outer(delta3, a2)
        Delta2 += np.outer(delta2, a1)

    # Bereken gemiddelde van de gradiënten
    Delta2_grad = Delta2 / m
    Delta3_grad = Delta3 / m

    return Delta2_grad, Delta3_grad

