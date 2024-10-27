import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class DataHandler:
    def __init__(self, url, column_names):
        self.url = url
        self.column_names = column_names
        self.data = self._load_data()
        self.scaler = StandardScaler()

    def _load_data(self):
        data = pd.read_csv(self.url, names=self.column_names)
        columns_to_replace = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        for column in columns_to_replace:
            data[column] = data[column].replace(0, data[column].mean())
        return data

    def preprocess_data(self):
        X = self.data.drop('Outcome', axis=1)
        y = self.data['Outcome']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        return X_train, X_test, y_train, y_test

    def scale_new_data(self, new_data):
        return self.scaler.transform(new_data)
