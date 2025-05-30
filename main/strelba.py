import numpy as np
import matplotlib.pyplot as plt

# Задаем параметры задачи
x0 = 0
xN = 1
h = 0.1
N = int((xN - x0) / h)
X = np.linspace(x0, xN, N + 1)

# Правая часть системы
def f(x, y1, y2):
    dy1dx = y2
    dy2dx = (2 * x * y2 - 2 * y1) / (x**2 + 1)
    return dy1dx, dy2dx

# Метод Рунге-Кутты 4-го порядка
def runge_kutta(eta, verbose=False):
    y1 = np.zeros(N + 1)
    y2 = np.zeros(N + 1)
    y1[0] = eta
    y2[0] = 1  # y'(0)

    if verbose:
        print(f"Начальные условия: y(0) = {eta:.6f}, y'(0) = {y2[0]:.6f}")
        print(f"{'Step':<4} {'x':<6} {'y':<10} {'y\'':<10}")
        print("-" * 32)
        print(f"{0:<4} {X[0]:<6.1f} {y1[0]:<10.6f} {y2[0]:<10.6f}")

    for i in range(N):
        x = X[i]
        k1_1, k1_2 = f(x, y1[i], y2[i])
        k2_1, k2_2 = f(x + h / 2, y1[i] + h * k1_1 / 2, y2[i] + h * k1_2 / 2)
        k3_1, k3_2 = f(x + h / 2, y1[i] + h * k2_1 / 2, y2[i] + h * k2_2 / 2)
        k4_1, k4_2 = f(x + h, y1[i] + h * k3_1, y2[i] + h * k3_2)

        y1[i + 1] = y1[i] + (h / 6) * (k1_1 + 2 * k2_1 + 2 * k3_1 + k4_1)
        y2[i + 1] = y2[i] + (h / 6) * (k1_2 + 2 * k2_2 + 2 * k3_2 + k4_2)

        if verbose:
            print(f"{i+1:<4} {X[i+1]:<6.1f} {y1[i+1]:<10.6f} {y2[i+1]:<10.6f}")

    return y1, y2

# Метод стрельбы (только два выстрела)
def shooting_method(eta1, eta2):
    print("="*60)
    print("МЕТОД СТРЕЛЬБЫ ДЛЯ КРАЕВОЙ ЗАДАЧИ")
    print("="*60)
    print("Дифференциальное уравнение: (x² + 1)y'' - 2xy' + 2y = 0")
    print("Граничные условия: y'(0) = 1, y(1) = 1")
    print("Неизвестное начальное условие: y(0) = η")
    print("="*60)
    
    print(f"\nПервый выстрел с η₁ = {eta1}")
    print("-" * 40)
    y1_1, y2_1 = runge_kutta(eta1, verbose=True)
    F1 = y1_1[-1] - 1  # y(1) - yN
    print(f"Результат: y(1) = {y1_1[-1]:.6f}, отклонение F₁ = {F1:.6f}")

    print(f"\nВторой выстрел с η₂ = {eta2}")
    print("-" * 40)
    y1_2, y2_2 = runge_kutta(eta2, verbose=True)
    F2 = y1_2[-1] - 1
    print(f"Результат: y(1) = {y1_2[-1]:.6f}, отклонение F₂ = {F2:.6f}")

    return (X, y1_1, y2_1, F1), (X, y1_2, y2_2, F2)

# Аналитическое решение
def analytical_solution(x):
    return x - x**2 + 1

# Запуск метода стрельбы с произвольными начальными eta1 и eta2 (не равными 1)
eta1 = 0.5
eta2 = 1.5

# Выполняем два выстрела
shot1, shot2 = shooting_method(eta1, eta2)

# Вычисляем аналитическое решение
Y_analytical = analytical_solution(X)

# Сравнение результатов
print(f"\n" + "="*60)
print("СРАВНЕНИЕ С АНАЛИТИЧЕСКИМ РЕШЕНИЕМ")
print("="*60)
print(f"Аналитическое решение: y(x) = x - x² + 1")
print(f"Точное начальное условие: y(0) = 1")

# Построение графика сравнения
plt.figure(figsize=(10, 6))
plt.plot(X, shot1[1], 'ro-', label=f'Численное решение η₁ = {eta1}', markersize=6, linewidth=2)
plt.plot(X, shot2[1], 'go-', label=f'Численное решение η₂ = {eta2}', markersize=6, linewidth=2)
plt.plot(X, Y_analytical, 'b--', label='Аналитическое решение', linewidth=3)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Сравнение численного и аналитического решений')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Итоговая сводка
print(f"\n" + "="*60)
print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
print("="*60)
print(f"Первый выстрел  (η₁ = {eta1}): y(1) = {shot1[1][-1]:.6f}, отклонение = {shot1[3]:.6f}")
print(f"Второй выстрел  (η₂ = {eta2}): y(1) = {shot2[1][-1]:.6f}, отклонение = {shot2[3]:.6f}")
print(f"Точное решение  (η = 1.0): y(1) = 1.000000, отклонение = 0.000000")
print("="*60)