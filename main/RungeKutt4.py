
"""
Решение задачи Коши для ОДУ второго порядка методом Рунге-Кутты 4-го порядка.
Уравнение:
    (x^2 + 1)y'' - 2xy' + 2y = 0
    y(0) = 1
    y'(0) = 1
    x ∈ [0, 1], h = 0.1

Преобразуем уравнение второго порядка к системе ОДУ первого порядка:

Обозначим:
    y1 = y
    y2 = y'

Тогда:
    y1' = y2
    y2' = ((2x)y2 - 2y1) / (x^2 + 1)

Мы выразили y'' из исходного уравнения:
    (x^2 + 1)y'' - 2xy' + 2y = 0
    => y'' = (2xy' - 2y) / (x^2 + 1)

Тогда система:
    dy1/dx = y2
    dy2/dx = (2x * y2 - 2 * y1) / (x^2 + 1)

Теперь можно применить Рунге-Кутту 4-го порядка.
"""

import numpy as np
import matplotlib.pyplot as plt

# Правая часть системы
def f(x, y):
    y1, y2 = y
    dy1 = y2
    dy2 = (2 * x * y2 - 2 * y1) / (x**2 + 1)
    return np.array([dy1, dy2])

def runge_kutta_4(f, y0, x0, x_end, h):
    x_values = [x0]
    y_values = [y0]
    
    x = x0
    y = y0

    print(f"Step-by-step RK4 iterations:")
    print(f"{'x':>6} | {'y1 (y)':>12} | {'y2 (dy/dx)':>12}")
    print("-" * 36)
    y_print = y.astype(float)  # Ensure y is a 1D array of floats
    print(f" {x:6.2f} | {y_print[0]:12.8f} | {y_print[1]:12.8f}")

    while x < x_end:
        k1 = h * f(x, y)
        k2 = h * f(x + h / 2, y + k1 / 2)
        k3 = h * f(x + h / 2, y + k2 / 2)
        k4 = h * f(x + h, y + k3)

        y = y + (k1 + 2*k2 + 2*k3 + k4) / 6
        x = round(x + h, 10)  # избежание накопления ошибки с плавающей точкой
        x_values.append(x)
        y_values.append(y)
        print(f"{x:6.2f} | {y[0]:12.8f} | {y[1]:12.8f}")

    return np.array(x_values), np.array(y_values)

# Условия задачи
x0 = 0
y0 = np.array([1, 1])  # y(0)=1, y'(0)=1
h = 0.1
x_end = 1
n_steps = int((x_end - x0) / h)

# Численное решение
x_vals, y_vals = runge_kutta_4(f, y0, x0, x_end, h)

# Точное решение: y = x - x^2 + 1
y_exact = x_vals - x_vals**2 + 1

# Погрешность
error = np.abs(y_vals[:, 0] - y_exact)
print(f"Максимальная погрешность: {np.max(error):.8f}")

# График решения
plt.figure(figsize=(10, 6))
plt.plot(x_vals, y_vals[:, 0], 'bo-', label='Численное решение (RK4)')
plt.plot(x_vals, y_exact, 'r--', label='Точное решение')
plt.title('Сравнение численного и точного решений')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.legend()
plt.show()

# График погрешности
plt.figure(figsize=(10, 4))
plt.plot(x_vals, error, 'm-o', label='Погрешность')
plt.title('Погрешность численного решения')
plt.xlabel('x')
plt.ylabel('|Численное - Точное|')
plt.grid(True)
plt.legend()
plt.show()
