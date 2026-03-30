from tensorflow.keras.callbacks import EarlyStopping
# from tensorflow.keras.optimizer import Adam
def train_model(model , X_train,Y_train,X_val,Y_val):
    model.compile(
    optimizer = 'adam',
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']
    )
    #early stopping
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience = 5,
        restore_best_weights = True 
    )

    # Train 
    history = model.fit(
        X_train , Y_train,
        epochs = 30,
        batch_size = 64,
        validation_data = (X_val , Y_val),
        callbacks = [early_stop],
        verbose = 1
    )