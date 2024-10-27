from DiabetesPredictionApp.diabetes_app import DiabetesApp

if __name__ == "__main__":
    app = DiabetesApp()
    app.run()
    app.predict_new_user()
