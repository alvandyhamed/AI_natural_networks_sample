# train.py
from model import DigitRecognizerModel

def train_and_save_model():
    # ساخت و آموزش مدل
    recognizer = DigitRecognizerModel()
    recognizer.build_model()
    recognizer.train_model()
    recognizer.save_model('digit_recognizer.h5')
    recognizer.print_model()
    recognizer.confusion_matrix()

if __name__ == "__main__":
    train_and_save_model()
