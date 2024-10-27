from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU
from tensorflow.keras.regularizers import l2
from keras.src.callbacks import EarlyStopping
from keras.src.optimizers import Adam

class DiabetesModel:
    def __init__(self, input_shape):
        self.model = self._build_model(input_shape)
        self.early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    def _build_model(self, input_shape):
        model = Sequential([
            Dense(32, input_shape=(input_shape,), kernel_regularizer=l2(0.01)),
            LeakyReLU(alpha=0.1),
            Dropout(0.3),
            Dense(16, kernel_regularizer=l2(0.01)),
            LeakyReLU(alpha=0.1),
            Dropout(0.3),
            Dense(8, kernel_regularizer=l2(0.01)),
            LeakyReLU(alpha=0.1),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train(self, X_train, y_train):
        history = self.model.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.1, callbacks=[self.early_stopping])
        return history

    def evaluate(self, X_test, y_test):
        loss, accuracy = self.model.evaluate(X_test, y_test)
        print(f"Model Accuracy: {accuracy * 100:.2f}%")
        return accuracy

    def predict(self, new_data):
        prediction = self.model.predict(new_data)
        return "دیابت دارد" if prediction[0] > 0.5 else "دیابت ندارد"
