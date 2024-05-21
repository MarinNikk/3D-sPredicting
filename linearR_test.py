import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

def readFile():
    Water = pd.read_csv('waterquality.csv').dropna().tail(200)

    return Water

def linReg(Water):
    print(Water)
    X = Water[['AirTemp (C)', 'WaterDepth (m)']]
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

    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')

    ax.scatter(Water['AirTemp (C)'], Water['WaterDepth (m)'], Water['WaterTemp (C)'],c='blue', marker='o')
    
    x_surf = np.linspace(Water['AirTemp (C)'].min(), Water['AirTemp (C)'].max(), 10)
    y_surf = np.linspace(Water['WaterDepth (m)'].min(), Water['WaterDepth (m)'].max(), 10)

    x_surf, y_surf = np.meshgrid(x_surf, y_surf)
    z_surf = model.predict(pd.DataFrame({'AirTemp (C)': x_surf.ravel(), 'WaterDepth (m)': y_surf.ravel()}))
    z_surf = z_surf.reshape(x_surf.shape)

    ax.plot_surface(x_surf, y_surf, z_surf, alpha = 0.5, color ="red")

    ax.set_xlabel('AirTemp (C)')
    ax.set_ylabel('WaterDepth (m)')
    ax.set_zlabel('WaterTemp (C)')

    dot_x, dot_y, dot_z = prediction(model, [21.11111111, 12])
    ax.scatter(dot_x,  dot_y, 27,c='green', marker='o')

    plt.show()
#11/5/2019,0,,6.5,9,12,27,21.11111111

def prediction(model, X_pred):
    y_pred = model.predict([X_pred])
    print('y_pred', y_pred)
    print('x_pred', X_pred)

    return X_pred[0], X_pred[1], y_pred


def main():
    linReg(readFile())

if __name__ == "__main__":
    main()

