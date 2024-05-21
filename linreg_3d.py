import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def readFile():
    Water = pd.read_csv('waterquality.csv').dropna().tail(200)

    salinity = Water['Salinity (ppt)']
    dissOx = Water['DissolvedOxygen (mg/L)']
    pH = Water['pH']
    secchiDepth = Water['SecchiDepth (m)']
    waterDepth = Water['WaterDepth (m)']
    waterTemperature = Water['WaterTemp (C)']
    airTemperature = Water['AirTemp (C)']

    print(Water.loc[:,['AirTemp (C)', 'WaterDepth (m)', 'WaterTemp (C)']])
    return airTemperature, waterDepth, waterTemperature

def readNumpy(pred1, pred2, target):

    PRED = np.vstack([np.ones(len(pred1)), pred1, pred2]).T
    beta, residuals, rank, s = np.linalg.lstsq(PRED, target, rcond=None)

    print("Intercept (beta_0):", beta[0])
    print("Coefficient for x (beta_1):", beta[1])
    print("Coefficient for y (beta_2):", beta[2])

    print("Residuals:", residuals)
    print("Rank of the matrix:", rank)

    #target pred1 pred2
    print("Singular values:", s)


    # Visualization
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(pred1, pred2, target, color='blue', label='Observed Data')

    #prediction
    mypred = predict(21.11111111, 12, beta)
    ax.scatter(21.1, 12, mypred,c='green', marker='o')


    x_plane = np.linspace(min(pred1), max(pred1), 10)
    y_plane = np.linspace(min(pred2), max(pred2), 10)
    X_plane, Y_plane = np.meshgrid(x_plane, y_plane)
    Z_plane = beta[0] + beta[1] * X_plane + beta[2] * Y_plane

    ax.plot_surface(X_plane, Y_plane, Z_plane, alpha=0.5, color='red', label='Regression Plane')

    ax.set_xlabel('Pred1 [AirTemp (C)]')
    ax.set_ylabel('Pred2 [WaterDepth (m)]')
    ax.set_zlabel('Target[WaterTemp (C)]')

    ax.legend()
    
    plt.show()

def predict(pred1, pred2, beta):
    prediction = beta[0] + beta[1] * pred1 + beta[2] * pred2
    print('prediction',prediction)
    return prediction

def main():
    pred1, pred2, target = readFile()
    readNumpy(pred1, pred2, target)

if __name__ == "__main__":
    main()