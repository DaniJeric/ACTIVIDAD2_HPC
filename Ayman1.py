import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import barycentric_interpolate, lagrange
from time import time

# Definimos las funciones
f1 = lambda x: np.sin(x)
f2 = lambda x: np.exp(-20 * x**2)
f3 = lambda x: 1 / (1 + 25 * x**2)

# Método de diferencias divididas de Newton
def newton_divided_diff(x, y):
    n = len(x)
    coeff = np.copy(y)
    for j in range(1, n):
        for i in range(n - 1, j - 1, -1):
            coeff[i] = (coeff[i] - coeff[i - 1]) / (x[i] - x[i - j])
    return coeff

def newton_polynomial(x, x_points, coeff):
    n = len(coeff)
    poly = coeff[-1]
    for i in range(n - 2, -1, -1):
        poly = poly * (x - x_points[i]) + coeff[i]
    return poly

# Parámetros
n_points = 21
x_vals = np.linspace(-1, 10, n_points)
x_plot = np.linspace(-1, 10, 5000)

functions = [f1, f2, f3]
function_names = ["sin(x)", "exp(-20*x^2)", "1/(1+25*x^2)"]

for func, name in zip(functions, function_names):
    print(f"\nProcesando la función {name}")

    # Valores de la función
    y_vals = func(x_vals)

    # Interpolación baricéntrica
    start_time = time()
    y_bary = barycentric_interpolate(x_vals, y_vals, x_plot)
    bary_time = time() - start_time

    # Interpolación con Newton
    start_time = time()
    coeff = newton_divided_diff(x_vals, y_vals)
    y_newton = newton_polynomial(x_plot, x_vals, coeff)
    newton_time = time() - start_time

    # Interpolación de Lagrange (SciPy)
    start_time = time()
    lagrange_poly = lagrange(x_vals, y_vals)  # Devuelve un polinomio
    y_lagrange = lagrange_poly(x_plot)       # Evalúa el polinomio en los puntos deseados
    lagrange_time = time() - start_time

    # Calcular errores absolutos
    y_true = func(x_plot)
    error_bary = np.abs(y_true - y_bary)
    error_newton = np.abs(y_true - y_newton)
    error_lagrange = np.abs(y_true - y_lagrange)

    # Graficar
    plt.figure(figsize=(10, 6))
    plt.plot(x_plot, y_true, label=f"{name} (Real)", linewidth=2)
    plt.plot(x_plot, y_bary, label="Interpolación Baricéntrica", linestyle="--")
    plt.plot(x_plot, y_newton, label="Interpolación Newton", linestyle="--")
    plt.plot(x_plot, y_lagrange, label="Interpolación Lagrange", linestyle="--")

    plt.scatter(x_vals, y_vals, color="red", label="Puntos de Interpolación")
    plt.title(f"Interpolación de {name}")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid()
    plt.show()

    # Mostrar resultados
    print(f"Tiempo Baricéntrico: {bary_time:.11f} s")
    print(f"Tiempo Newton: {newton_time:.6f} s")
    print(f"Tiempo Lagrange: {lagrange_time:.6f} s")
    print(f"Error medio absoluto Baricéntrico: {np.mean(error_bary):.6e}")
    print(f"Error medio absoluto Newton: {np.mean(error_newton):.6e}")
    print(f"Error medio absoluto Lagrange: {np.mean(error_lagrange):.6e}")