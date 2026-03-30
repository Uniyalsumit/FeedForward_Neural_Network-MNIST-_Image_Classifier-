import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix 
def plot_history(history):
    #loss
    plt.plot(history.history['loss'],label = 'Train - loss')
    plt.plot(history.history['val_loss'], label = 'Validation Loss')
    plt.legend()
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("loss")
    plt.show()

    #Accuracy
    plt.plot(history.history['accuracy'], label = 'Train Accuracy')
    plt.plot(history.history['val_accuracy'] , label = 'Validation Accuracy')
    plt.legend()
    plt.title("Accuracy curve")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.show()

  
