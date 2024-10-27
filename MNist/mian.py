import numpy as np
from PIL import Image
from keras.src.datasets import mnist
from matplotlib import pyplot as plt

(X_train, y_train), (X_test, y_test) = mnist.load_data()

print(f"شکل داده‌های آموزشی: {X_train.shape}, شکل برچسب‌ها: {y_train.shape}")
print(f"شکل داده‌های تست: {X_test.shape}, شکل برچسب‌ها: {y_test.shape}")

# نمایش چند نمونه از تصاویر
plt.figure(figsize=(10, 5))
for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.imshow(X_train[i], cmap='gray')
    plt.title(f"Label: {y_train[i]}")
    plt.axis('off')
plt.show()

# نرمال‌سازی داده‌ها
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

# تغییر شکل داده‌ها به (ارتفاع، عرض، کانال)
X_train = X_train.reshape((X_train.shape[0], 28, 28, 1))
X_test = X_test.reshape((X_test.shape[0], 28, 28, 1))

print(f"شکل جدید داده‌های آموزشی: {X_train.shape}")

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# ساخت مدل CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),  # لایه کانولوشن اول
    MaxPooling2D((2, 2)),  # لایه Max Pooling اول
    Conv2D(64, (3, 3), activation='relu'),  # لایه کانولوشن دوم
    MaxPooling2D((2, 2)),  # لایه Max Pooling دوم
    Flatten(),  # صاف کردن داده‌ها برای لایه‌های پنهان
    Dense(64, activation='relu'),  # لایه پنهان
    Dense(10, activation='softmax')  # لایه خروجی با 10 نرون (برای اعداد 0 تا 9)
])

# کامپایل کردن مدل
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# نمایش خلاصه مدل
model.summary()
# آموزش مدل
history = model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.1)
loss, accuracy = model.evaluate(X_test, y_test)
print(f"دقت مدل روی داده‌های تست: {accuracy * 100:.2f}%")

# بارگذاری تصویر جدید
image_path = "your_image.png"  # مسیر تصویر جدید
image = Image.open(image_path).convert('L')  # تبدیل به سیاه و سفید
image = image.resize((28, 28))  # تغییر اندازه به 28x28

# تبدیل تصویر به آرایه و آماده‌سازی برای مدل
image_array = np.array(image)
image_array = image_array.astype('float32') / 255  # نرمال‌سازی
image_array = image_array.reshape(1, 28, 28, 1)  # تغییر شکل برای ورودی مدل

# پیش‌بینی با مدل
prediction = model.predict(image_array)
predicted_label = np.argmax(prediction)

print(f"عدد پیش‌بینی شده: {predicted_label}")