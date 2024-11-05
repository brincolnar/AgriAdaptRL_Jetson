import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import Polynomial

def poly_fit(x, y, degree=2):
    """ Fits a polynomial of given degree to the data """
    coefs = Polynomial.fit(x, y, degree).convert().coef
    return coefs

x = np.array([10, 50, 100])

''' old 
y_75_balance = np.array([1, -0.3, -1])
y_50_balance = np.array([-0.5, 0.7, -0.5])
y_25_balance = np.array([-1, 0.4, 1.3])
y_100_balance = np.array([-1, 0.3, 1.5])

y_75_accuracy = np.array([0.3, -0.3, -1])
y_50_accuracy = np.array([-0.2, 0.5, 0.3])
y_25_accuracy = np.array([-0.5, 0.4, 1.3])
y_100_accuracy = np.array([-0.7, 0.4, 3])

y_75_extreme_accuracy = np.array([-1, -0.3, 0.3])
y_50_extreme_accuracy = np.array([0.3, 0.5, 0.3])
y_25_extreme_accuracy = np.array([-0.5, 0.4, 1.3])
y_100_extreme_accuracy = np.array([0.3, 1, 1])

y_75_energy = np.array([1, 0.5, 0])
y_50_energy = np.array([-0.5, 0.7, 0.9])
y_25_energy = np.array([-1, 0.4, 1.3])
y_100_energy = np.array([-1, 0.2, 1])
'''

y_75_balance = np.array([1, -0.3, -1])
y_50_balance = np.array([-0.5, 0.7, -0.5])
y_25_balance = np.array([-1, 0.4, 1.3])
y_100_balance = np.array([-1, 0.3, 1.5])

y_75_accuracy = np.array([0.3, 0.2, 0.0 ])
y_50_accuracy = np.array([0.6, 0.8, 0.5])
y_25_accuracy = np.array([0.5, 0.7, 0.9])
y_100_accuracy = np.array([0.4, 0.7, 1])

y_75_extreme_accuracy = np.array([0.3, 0.2, 0.05])
y_50_extreme_accuracy = np.array([0.6, 0.8, 0.5])
y_25_extreme_accuracy = np.array([0.4, 0.6, 0.8])
y_100_extreme_accuracy = np.array([0.5, 0.8, 1])

y_75_energy = np.array([1, 0.5, 0])
y_50_energy = np.array([-0.5, 0.7, 0.9])
y_25_energy = np.array([-1, 0.4, 1.3])
y_100_energy = np.array([-1, 0.2, 1])

y_75_extreme_energy = np.array([1, 1.0, 0.8])
y_50_extreme_energy = np.array([0, 0.4, 0.6])
y_25_extreme_energy = np.array([-0.5, 0, 0.3])
y_100_extreme_energy = np.array([-1, -0.3, 0.3])


# Fit a polynomial to each dataset
# coefs25 = poly_fit(x, y_25_extreme_accuracy)
# coefs50 = poly_fit(x, y_50_extreme_accuracy)
# coefs75 = poly_fit(x, y_75_extreme_accuracy)
# coefs100 = poly_fit(x, y_100_extreme_accuracy)

# coefs25 = poly_fit(x, y_25_accuracy)
# coefs50 = poly_fit(x, y_50_accuracy)
# coefs75 = poly_fit(x, y_75_accuracy)
# coefs100 = poly_fit(x, y_100_accuracy)

# coefs25 = poly_fit(x, y_25_energy)
# coefs50 = poly_fit(x, y_50_energy)
# coefs75 = poly_fit(x, y_75_energy)
# coefs100 = poly_fit(x, y_100_energy)

coefs25 = poly_fit(x, y_25_extreme_energy)
coefs50 = poly_fit(x, y_50_extreme_energy)
coefs75 = poly_fit(x, y_75_extreme_energy)
coefs100 = poly_fit(x, y_100_extreme_energy)


x_model = np.linspace(0, 100, 400)
y_model_25 = np.polyval(coefs25[::-1], x_model)
y_model_50 = np.polyval(coefs50[::-1], x_model)
y_model_75 = np.polyval(coefs75[::-1], x_model)
y_model_100 = np.polyval(coefs100[::-1], x_model)


plt.figure(figsize=(10, 8))
plt.scatter(x, y_25_energy, color='blue', label='Data 25%')
plt.plot(x_model, y_model_25, label='Fit 25%', color='blue')
plt.scatter(x, y_50_energy, color='red', label='Data 50%')
plt.plot(x_model, y_model_50, label='Fit 50%', color='red')
plt.scatter(x, y_75_energy, color='green', label='Data 75%')
plt.plot(x_model, y_model_75, label='Fit 75%', color='green')
plt.scatter(x, y_100_energy, color='purple', label='Data 100%')
plt.plot(x_model, y_model_100, label='Fit 100%', color='purple')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Quadratic Polynomial Fit for Different Networks')
plt.ylim(-1.5, 1.5)
plt.legend()
plt.savefig('./fit.png')