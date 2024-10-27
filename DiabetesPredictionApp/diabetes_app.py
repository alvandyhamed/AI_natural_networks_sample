from data.data_handler import DataHandler
from model.diabetes_model import DiabetesModel
from utils.plotter import plot_metrics
import numpy as np

class DiabetesApp:
    def __init__(self):
        self.data_handler = DataHandler(
            url="https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv",
            column_names=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
        )
        X_train, X_test, y_train, y_test = self.data_handler.preprocess_data()
        self.model = DiabetesModel(input_shape=X_train.shape[1])
        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test

    def run(self):
        history = self.model.train(self.X_train, self.y_train)
        self.model.evaluate(self.X_test, self.y_test)
        plot_metrics(history)

    def predict_new_user(self):
        pregnancies = float(input("\nتعداد بارداری‌ها: "))
        glucose = float(input("\nمقدار گلوکز: "))
        blood_pressure = float(input("\nفشار خون: "))
        skin_thickness = float(input("\nضخامت پوست: "))
        insulin = float(input("\nانسولین: "))
        bmi = float(input("\nشاخص توده بدنی (BMI): "))
        dpf = float(input("Diabetes Pedigree Function: \n"))
        age = float(input("\nسن: "))

        new_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
        new_data_scaled = self.data_handler.scale_new_data(new_data)
        result = self.model.predict(new_data_scaled)
        print(f"نتیجه پیش‌بینی: {result}")
