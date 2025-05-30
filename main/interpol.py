import numpy as np
import matplotlib.pyplot as plt
from math import factorial
from sympy import symbols, diff, sin, lambdify, simplify

# Исходная функция
def f(x):
    return np.sin(x) + x

# Узлы интерполяции
X = np.array([0, np.pi/6, 5*np.pi/12, np.pi/2])
Y = f(X)

# Точка для проверки
X_star = np.pi / 4
true_value = f(X_star)

# 1. Лагранж
def lagrange_interp(x, X, Y):
    n = len(X)
    L = 0
    for i in range(n):
        term = Y[i]
        for j in range(n):
            if j != i:
                term *= (x - X[j]) / (X[i] - X[j])
        L += term
    return L

# 2. Ньютон
def divided_diff(X, Y):
    n = len(Y)
    coef = np.copy(Y)
    for j in range(1, n):
        coef[j:n] = (coef[j:n] - coef[j - 1:n - 1]) / (X[j:n] - X[0:n - j])
    return coef

def newton_interp(x, X, coef):
    n = len(X)
    result = coef[0]
    product = 1.0
    for i in range(1, n):
        product *= (x - X[i - 1])
        result += coef[i] * product
    return result

# 3. Вычисление
lag_val = lagrange_interp(X_star, X, Y)
newton_coef = divided_diff(X, Y)
newton_val = newton_interp(X_star, X, newton_coef)

# 4. Погрешности
error_lagrange = abs(true_value - lag_val)
error_newton = abs(true_value - newton_val)

# 5. Аналитическая оценка погрешности Лагранжа
# Остаточный член: R(x) = f^{(n)}(ξ)/(n!) * (x - x0)...(x - xn)
x_sym = symbols('x')
f_sym = sin(x_sym) + x_sym
f4_sym = diff(f_sym, x_sym, 4)
f4_func = lambdify(x_sym, f4_sym)

# Оценим максимум производной 4-го порядка на отрезке [0, pi/2]
grid = np.linspace(0, np.pi/2, 1000)
f4_vals = np.abs(f4_func(grid))
f4_max = np.max(f4_vals)

# Вычисляем остаточный член
omega = np.prod([X_star - xi for xi in X])
n = len(X)
R_lagrange = f4_max / factorial(n) * abs(omega)

# === Вывод результатов ===
print("=== Значения интерполяции в X* = π/4 ===")
print(f"Истинное значение:            {true_value:.10f}")
print(f"Интерполяция Лагранжа:        {lag_val:.10f}")
print(f"Интерполяция Ньютона:         {newton_val:.10f}")
print()
print("=== Погрешности ===")
print(f"Погрешность Лагранжа:         {error_lagrange:.2e}")
print(f"Погрешность Ньютона:          {error_newton:.2e}")
print()
print("=== Аналитическая проверка ===")
print(f"f⁽⁴⁾(x) = {simplify(f4_sym)}")
print(f"Максимум |f⁽⁴⁾(x)| на [0, π/2]: {f4_max:.6f}")
print(f"|ω(x)| = |Π(x* - xi)| = {abs(omega):.6f}")
print(f"Оценка остаточного члена:     {R_lagrange:.2e}")
print()

# 6. Графики
x_vals = np.linspace(0, np.pi/2, 400)
f_vals = f(x_vals)
lag_vals = [lagrange_interp(x, X, Y) for x in x_vals]
newt_vals = [newton_interp(x, X, newton_coef) for x in x_vals]

plt.figure(figsize=(10, 6))
plt.plot(x_vals, f_vals, label='y = sin(x) + x', linewidth=2)
plt.plot(x_vals, lag_vals, '--', label='Lagrange Polynomial', linewidth=2)
plt.plot(x_vals, newt_vals, ':', label='Newton Polynomial', linewidth=2)
plt.plot(X, Y, 'o', label='Interpolation Points')
plt.axvline(X_star, color='gray', linestyle='--', label='X* = π/4')
plt.scatter([X_star], [true_value], color='black', zorder=5, label='True y(π/4)')
plt.title("Интерполяция функции y = sin(x) + x")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
