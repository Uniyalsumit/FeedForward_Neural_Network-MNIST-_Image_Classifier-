import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
def load_and_preprocess():
    #loading dataset
    (X_train , Y_train),(X_test,Y_test) = mnist.load_data()
    #Normalize (0-255 -> 0-1)
    X_train = X_train/255.0
    X_test = X_test/255.0

    #flatten image
    X_train = X_train.reshape(X_train.shape[0],-1)
    X_test = X_test.reshape(X_test.shape[0],-1)
    Y_train = Y_train.flatten()
    Y_test = Y_test.flatten()
    # One hot encode labels 
    Y_train = to_categorical(Y_train,10)
    Y_test = to_categorical(Y_test,10)
    #Splittig dataset
    X_train ,X_val,Y_train,Y_val = train_test_split(X_train,Y_train,test_size =0.2 ,random_state = 42)

