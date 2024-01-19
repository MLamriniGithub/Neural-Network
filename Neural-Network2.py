"""
Main.py

The main script


In this file, we explain the various steps of creating, training, and testing a Neural Network (NN) in Python. The modules used include:
sklearn for data preparation and keras/tensorflow for the actual NN manipulation.
The dataset used in this guide is Iris. The goal is to predict the Species class.

"""
import pandas as pn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

#1. Data Preparation

#In the case of a multiclassification problem with m classes, we transform
#The class modalities into m-tuples with a single value of 1 and all
#other values set to 0.
def encodeClass(s_class):
    if (s_class=='Iris-setosa'):
        return [1,0,0]
    elif (s_class=='Iris-versicolor'):
        return [0,1,0]
    else:
        return [0,0,1]
		
#Data Reading
#myData = pd.read_csv('iris.csv', sep=',')
myData=pn.read_csv('iris.csv', sep=',')

#Extraction of some features from the dataset
nbColumns = len(myData.columns)
classes = myData['species'].unique().tolist()
nbClasses = len(classes)

X=myData.values[:,:nbColumns-1]
X=X.astype('float64')
Y=myData.values[:,nbColumns-1]

#Encoding the classes.
encoded_Y = np.array([encodeClass(y) for y in list(Y)])

#Creating training and test sets.
X_train, X_test, Y_train, Y_test = train_test_split( X, encoded_Y, test_size = 0.3, random_state = 100)

#2. Building the Neural Network
#Creating a 'blank' network: Using the Sequential() function. This function is located in Keras
nn = Sequential()

#Adding layers one by one. The Keras module provides a function for each type of layer. 
#Here, we exclusively use the dense function corresponding to the layers of the standard neural network.
#To add a layer, specify its number of neurons and activation function. 
#For the first hidden layer, also specify the number of inputs (i.e., the number of neurons in the input layer).

nn.add(Dense(5, input_dim=nbColumns-1, activation='sigmoid'))
nn.add(Dense(nbClasses, activation='softmax'))

nn.summary()

#Complete the specifications of the Neural Network, including the loss function 
#(here, categorical_crossentropy as it is a multiclassification problem) and the metric used to measure its performance (here, accuracy).
nn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

nn.fit(X_train, Y_train, epochs=200, batch_size=10)

#4. Test 
#accuracy

score = nn.evaluate(X_test, Y_test, verbose=0)
print('Test accuracy:', score[1])

Y_pred = nn.predict(X_test)
Y_pred_1 = Y_pred.argmax(axis=1)
Y_test_1 = Y_test.argmax(axis=1)
confusion = confusion_matrix(Y_pred_1, Y_test_1)
print(confusion) 

 