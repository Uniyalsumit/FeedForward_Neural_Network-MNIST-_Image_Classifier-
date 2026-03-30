from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
def evaluate_model(model,X_test,Y_test):
    #Evalution
    test_loss , test_acc = model.evaluate(X_test, Y_test)
    print("Test Accuracy : ",test_acc)

    #predictions
    Y_pred = model.predict(X_test)
    Y_pred_class = np.argmax(Y_pred , axis = 1)
    Y_true = np.argmax(Y_test,axis =1)

    # Metrics
    print(classification_report(Y_true,Y_pred_class))

    #Confusion Matrix
    cm = confusion_matrix(Y_true,Y_pred_class)
    print(cm)
    #heatmap
    plt.figure()
    sns.heatmap(cm,annot=True,fmt = 'd')
    plt.title("Confusion Matrix")
    plt.xlabel("predicted")
    plt.ylabel("Actual")
    plt.show()