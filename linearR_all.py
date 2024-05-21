import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

def readFile():
    Water = pd.read_csv('waterquality.csv').dropna().tail(200)

    return Water

# 'Salinity (ppt)', 'DissolvedOxygen (mg/L)', 'pH', 'SecchiDepth (m)', 'WaterDepth (m)', 'AirTemp (C)'

def linReg(Water):
    print(Water)
    X = Water[['Salinity (ppt)', 'DissolvedOxygen (mg/L)', 'pH', 'SecchiDepth (m)', 'WaterDepth (m)', 'AirTemp (C)']]
    y = Water['WaterTemp (C)']
    # Create a linear regression model
    model = LinearRegression()

    # Train the model
    model.fit(X, y)

    # Make predictions
    y_pred = model.predict(X)

    # Display the learning outcome
    print("\nLearning Outcome:")
    print(f"Coefficients: {model.coef_}")
    print(f"Intercept: {model.intercept_}")
    print(f"Mean squared error (MSE): {mean_squared_error(y, y_pred)}")
    print(f"Coefficient of determination (R^2): {r2_score(y, y_pred)}")


def main():
    linReg(readFile())

if __name__ == "__main__":
    main()

