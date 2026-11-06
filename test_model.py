
import joblib
import numpy as np
import pandas as pd

def test_model_prediction():
    model = joblib.load('diabetes_model.joblib')

    x = pd.DataFrame(
        [[0.02, 0.07, 0.03, 0.02, 0.01,
          0.03, 0.01, 0.04, 0.03, 0.01]],
        columns=model.feature_names_in_
    )

    prediction = model.predict(x)

    assert isinstance(prediction[0], (float, np.floating))


def test_model_output_shape():
    model = joblib.load('diabetes_model.joblib')

    x = pd.DataFrame(
        np.zeros((5, 10)),
        columns=model.feature_names_in_
    )

    prediction = model.predict(x)

    assert prediction.shape == (5,)
