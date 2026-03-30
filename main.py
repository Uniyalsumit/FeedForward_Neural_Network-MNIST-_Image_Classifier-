import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np

from evaluate import evaluate_model
from model import build_model
from preprocess import load_and_preprocess
from train import train_model
from utils import plot_history
def main():
    X_train, X_val, X_test, Y_train, Y_val, Y_test = load_and_preprocess()
   
    model = build_model()

    history = train_model(model, X_train,Y_train,X_val,X_val)

    plot_history(history)

    evaluate_model(model,X_test,Y_test)
    model.save("outputs/fnn_mnist_model.h5")

if __name__ == "__main__":
    main()