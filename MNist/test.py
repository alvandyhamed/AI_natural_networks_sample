import cv2
import numpy as np
import tensorflow as tf


# بارگذاری مدل ذخیره شده
model = tf.keras.models.load_model("mnist_cnn_model.h5")


# تابع برای پیش‌بینی عدد از روی تصویر
def predict_digit(image_path):
    # بارگذاری تصویر و تبدیل به مقیاس خاکستری
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # تغییر اندازه تصویر به 28x28 پیکسل
    img = cv2.resize(img, (28, 28))

    # معکوس کردن رنگ‌ها (در صورت نیاز)
    img = cv2.bitwise_not(img)

    # نرمال‌سازی تصویر
    img = img.astype("float32") / 255.0

    # تغییر شکل تصویر برای ورودی مدل
    img = np.expand_dims(img, axis=-1)  # اضافه کردن بعد کانال
    img = np.expand_dims(img, axis=0)  # اضافه کردن بعد batch

    # پیش‌بینی عدد
    prediction = model.predict(img)
    predicted_digit = np.argmax(prediction)

    return predicted_digit


# استفاده از تابع برای پیش‌بینی
image_path = "your_image.png"  # مسیر تصویر مورد نظر
predicted_digit = predict_digit(image_path)
print(f"The predicted digit is: {predicted_digit}")
