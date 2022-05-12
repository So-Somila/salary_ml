# %%
from pyexpat import model
from tkinter import Y
from turtle import color
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


salaries = [10, 20, 30, 50, 40, 60, 65, 100, 150, 120, 160, 200]
months = []

for i in range(1, len(salaries) + 1):
    months.append(i)


def predict_salary(months_experience):
    raw_data = {
        'months': months,
        'salary': salaries
    }

    df = pd.DataFrame(raw_data)

    # get the data we want to predict
    X = np.array(df['months']).reshape(-1, 1)
    y = np.array(df['salary']).reshape(-1, 1)

    # split the testing data and training data
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=.40)

    # Initialise the model
    model = LinearRegression()
    model.fit(train_X, train_y)

    # make the prediction
    y_prediction = model.predict([[months_experience]])
    print('PREDICTION:', round(y_prediction[0][0]), '$ per month')

    # used to test how accurate the model is
    y_test_prediction = model.predict(test_X)
    y_line = model.predict(X)

    # get the slope of the model
    slope = model.coef_[0][0]
    mae = mean_absolute_error(test_y, y_test_prediction)
    r2 = r2_score(test_y, y_test_prediction)

    print("Slope: +", round(slope), "$ per month")
    print("MAE +-", round(mae, 2))
    print("r2:", round(r2, 2) * 100, "% accurate.")

    plt.scatter(X, y, s=12)
    plt.xlabel('Months')
    plt.ylabel('Salary')
    plt.plot(X, y_line, color='r')
    plt.show


predict_salary(14)

# %%
