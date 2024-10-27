import cv2
from model import DigitRecognizerModel
import numpy as np

class DigitCamera:
    def __init__(self, model_path='mnist_cnn_model.h5'):
        self.recognizer = DigitRecognizerModel()
        self.recognizer.load_model(model_path)

    def run_camera(self):
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # تبدیل تصویر به مقیاس خاکستری
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

            # پیدا کردن کانتورها
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)

                # فیلتر کردن نواحی کوچک و بزرگ غیر منطقی
                if w < 40 or h < 40 or w > 300 or h > 300:
                    continue

                # تغییر اندازه تصویر به 28x28 و نرمال‌سازی
                digit = gray[y:y + h, x:x + w]
                digit_resized = cv2.resize(digit, (28, 28))
                digit_resized = digit_resized.astype('float32') / 255.0

                # تبدیل تصویر به فرمت ورودی مدل (1, 28, 28, 1)
                digit_reshaped = np.expand_dims(digit_resized, axis=0)
                digit_reshaped = np.expand_dims(digit_reshaped, axis=-1)

                # پیش‌بینی عدد
                predicted_label = self.recognizer.predict_digit(digit_reshaped)

                # رسم مستطیل و نمایش عدد پیش‌بینی شده
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, str(predicted_label), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # نمایش تصویر در پنجره
            cv2.imshow("Digits Recognition", frame)

            # توقف با فشار دادن 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # آزاد کردن منابع
        cap.release()
        cv2.destroyAllWindows()
