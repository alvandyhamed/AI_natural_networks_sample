# main.py
from model import DigitRecognizerModel
from camera import DigitCamera


def main():


    # باز کردن دوربین و شناسایی اعداد
    camera = DigitCamera('digit_recognizer.h5')
    #camera = DigitCamera('cnn-mnist-model.h5')
    camera.run_camera()


if __name__ == "__main__":
    main()
