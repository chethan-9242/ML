
import pandas as pd
import joblib

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

data = load_diabetes()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Original MSE:", mean_squared_error(y_test, y_pred))

joblib.dump(model, "diabetes_model.joblib")

loaded_model = joblib.load("diabetes_model.joblib")

loaded_pred = loaded_model.predict(X_test)

print("Loaded Model MSE:", mean_squared_error(y_test, loaded_pred))
