import tensorflow as tf
from keras.src.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.src.layers import BatchNormalization
from keras.src.optimizers import Adam
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.src.datasets import mnist
import numpy as np
from keras.src.legacy.preprocessing.image import ImageDataGenerator
import seaborn as sns
from tensorflow.keras.regularizers import l2


class DigitRecognizerModel:
    def __init__(self):
        self.model = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

    def build_model(self):
        self.model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.3),
            Conv2D(64, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.3),
            Conv2D(128, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(10, activation='softmax')
        ])

        self.model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

    def train_model(self):
        # بارگذاری و پیش‌پردازش داده‌های MNIST
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        X_train = X_train.astype('float32') / 255
        X_train = X_train.reshape((X_train.shape[0], 28, 28, 1))

        # پیش‌پردازش داده‌های تست
        X_test = X_test.astype('float32') / 255
        X_test = X_test.reshape((X_test.shape[0], 28, 28, 1))

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        # تعریف Data Augmentation
        datagen = ImageDataGenerator(
            rotation_range=40,  # تغییر زاویه تا 30 درجه
            width_shift_range=0.3,
            height_shift_range=0.3,
            shear_range=0.3,
            zoom_range=0.3,
            brightness_range=[0.4, 1.8],  # تغییر روشنایی
            horizontal_flip=True
        )
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.00001)
        early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

        # self.model.fit(datagen.flow(X_train, y_train, batch_size=64),
        #                epochs=10, validation_data=(X_test, y_test),
        #                callbacks=[reduce_lr])
        self.model.fit(datagen.flow(X_train, y_train, batch_size=64),
                       epochs=30, validation_data=(X_test, y_test),
                       callbacks=[reduce_lr, early_stopping])

    def save_model(self, path='digit_recognizer.h5'):
        self.model.save(path)

    def load_model(self, path='digit_recognizer.h5'):
        self.model = tf.keras.models.load_model(path)
        # کامپایل مجدد با تنظیمات دقیق
        self.model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

    def print_model(self):
        y_pred = self.model.predict(self.X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        cm = confusion_matrix(self.y_test, y_pred_classes)

        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()

    def confusion_matrix(self):
        y_true = self.y_test  # برچسب‌های واقعی
        y_pred_probabilities = self.model.predict(self.X_test)  # پیش‌بینی‌های مدل
        y_pred = np.argmax(y_pred_probabilities, axis=1)  # تبدیل احتمال‌ها به برچسب

        # محاسبه Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)

        # رسم نمودار Confusion Matrix
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()

    def predict_digit(self, image):
        image = np.array(image).astype('float32') / 255
        image = image.reshape(1, 28, 28, 1)
        prediction = self.model.predict(image)
        return np.argmax(prediction)
