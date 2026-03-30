from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense , Input,Dropout

def build_model():
    model = Sequential([
    Input(shape=(784,)),
    Dense(512 , activation = 'relu'),
    Dropout(0.5), 
    Dense(256 , activation = 'relu'),
     Dropout(0.4), 
    Dense(128 , activation = 'relu'),
     Dropout(0.3), 
    Dense(10 , activation = 'softmax')
])