import numpy as np
import matplotlib.pyplot as plt
import math

def f(x):
    """Функция f(x) = 3^x - 5x^2 + 1"""
    return 3**x - 5*x**2 + 1

def df(x):
    """Производная f'(x) = 3^x * ln(3) - 10x"""
    return 3**x * math.log(3) - 10*x

def plot_function():
    """Построение графика функции для локализации корней"""
    print("=" * 60)
    print("ГРАФИЧЕСКАЯ ЛОКАЛИЗАЦИЯ КОРНЕЙ")
    print("=" * 60)
    
    # Создаем массив значений x
    x = np.linspace(-1, 3, 1000)
    y = []
    
    # Вычисляем значения функции
    for xi in x:
        try:
            yi = f(xi)
            y.append(yi)
        except OverflowError:
            y.append(float('inf'))
    
    y = np.array(y)
    
    # Строим график
    plt.figure(figsize=(12, 8))
    plt.plot(x, y, 'b-', linewidth=2, label='f(x) = 3^x - 5x^2 + 1')
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.7, label='y = 0')
    plt.grid(True, alpha=0.3)
    plt.xlabel('x', fontsize=12)
    plt.ylabel('f(x)', fontsize=12)
    plt.title('График функции f(x) = 3^x - 5x^2 + 1', fontsize=14)
    plt.legend(fontsize=12)
    plt.ylim(-10, 20)
    
    # Находим приблизительные интервалы с корнями
    intervals = []
    for i in range(len(x)-1):
        if not (np.isinf(y[i]) or np.isinf(y[i+1])):
            if y[i] * y[i+1] < 0:
                intervals.append((x[i], x[i+1]))
                plt.plot([x[i], x[i+1]], [0, 0], 'ro', markersize=8)
    
    plt.show()
    
    print(f"Найдено интервалов с корнями: {len(intervals)}")
    for i, (a, b) in enumerate(intervals):
        print(f"Интервал {i+1}: [{a:.3f}, {b:.3f}]")
        print(f"  f({a:.3f}) = {f(a):.6f}")
        print(f"  f({b:.3f}) = {f(b):.6f}")
    
    # Находим наименьший положительный интервал
    positive_intervals = [(a, b) for a, b in intervals if b > 0]
    if positive_intervals:
        min_positive = min(positive_intervals, key=lambda x: x[0] if x[0] > 0 else x[1])
        print(f"\nНаименьший положительный интервал: [{min_positive[0]:.3f}, {min_positive[1]:.3f}]")
        return min_positive
    else:
        print("Положительных корней не найдено в рассматриваемой области")
        return None

def bisection_method(a, b, tolerance=0.001, max_iter=100):
    """Метод дихотомии (деления пополам)"""
    print("\n" + "=" * 60)
    print("МЕТОД ДИХОТОМИИ")
    print("=" * 60)
    print(f"Начальный интервал: [{a:.6f}, {b:.6f}]")
    print(f"Точность: {tolerance}")
    print()
    
    if f(a) * f(b) > 0:
        print("Ошибка: функция не меняет знак на концах интервала")
        return None
    
    print(f"{'№':>3} {'a':>12} {'b':>12} {'c':>12} {'f(c)':>12} {'|b-a|':>12}")
    print("-" * 75)
    
    iteration = 0
    while abs(b - a) > tolerance and iteration < max_iter:
        c = (a + b) / 2
        fc = f(c)
        
        print(f"{iteration+1:>3} {a:>12.6f} {b:>12.6f} {c:>12.6f} {fc:>12.6f} {abs(b-a):>12.6f}")
        
        if abs(fc) < tolerance:
            print(f"\nНайден корень с нужной точностью: x = {c:.6f}")
            return c
        
        if f(a) * fc < 0:
            b = c
        else:
            a = c
        
        iteration += 1
    
    root = (a + b) / 2
    print(f"\nРешение методом дихотомии: x = {root:.6f}")
    print(f"Количество итераций: {iteration}")
    print(f"f({root:.6f}) = {f(root):.8f}")
    return root

def newton_method(x0, tolerance=0.001, max_iter=100):
    """Метод Ньютона"""
    print("\n" + "=" * 60)
    print("МЕТОД НЬЮТОНА")
    print("=" * 60)
    print(f"Начальное приближение: x₀ = {x0:.6f}")
    print(f"Точность: {tolerance}")
    print()
    
    print(f"{'№':>3} {'x_n':>12} {'f(x_n)':>12} {'f\\(x_n)':>12} {'x_{{n+1}}':>12} {'|Δx|':>12}")
    print("-" * 75)
    
    x = x0
    for iteration in range(max_iter):
        fx = f(x)
        dfx = df(x)
        
        if abs(dfx) < 1e-12:
            print("Ошибка: производная близка к нулю")
            return None
        
        x_new = x - fx / dfx
        error = abs(x_new - x)
        
        print(f"{iteration+1:>3} {x:>12.6f} {fx:>12.6f} {dfx:>12.6f} {x_new:>12.6f} {error:>12.6f}")
        
        if error < tolerance:
            print(f"\nРешение методом Ньютона: x = {x_new:.6f}")
            print(f"Количество итераций: {iteration + 1}")
            print(f"f({x_new:.6f}) = {f(x_new):.8f}")
            return x_new
        
        x = x_new
    
    print("Превышено максимальное количество итераций")
    return x

def iteration_method(x0, tolerance=0.001, max_iter=100):
    """Метод простой итерации"""
    print("\n" + "=" * 60)
    print("МЕТОД ПРОСТОЙ ИТЕРАЦИИ")
    print("=" * 60)
    
    # Преобразуем уравнение 3^x - 5x^2 + 1 = 0 к виду x = g(x)
    # Один из способов: 3^x + 1 = 5x^2, откуда x = sqrt((3^x + 1)/5)
    def g(x):
        try:
            val = (3**x + 1) / 5
            if val < 0:
                return x  # возвращаем текущее значение если под корнем отрицательное число
            return math.sqrt(val)
        except:
            return x
    
    print(f"Итерационная формула: x = √((3^x + 1)/5)")
    print(f"Начальное приближение: x₀ = {x0:.6f}")
    print(f"Точность: {tolerance}")
    print()
    
    print(f"{'№':>3} {'x_n':>12} {'g(x_n)':>12} {'x_{n+1}':>12} {'|Δx|':>12}")
    print("-" * 65)
    
    x = x0
    for iteration in range(max_iter):
        gx = g(x)
        x_new = gx
        error = abs(x_new - x)
        
        print(f"{iteration+1:>3} {x:>12.6f} {gx:>12.6f} {x_new:>12.6f} {error:>12.6f}")
        
        if error < tolerance:
            print(f"\nРешение методом итерации: x = {x_new:.6f}")
            print(f"Количество итераций: {iteration + 1}")
            print(f"f({x_new:.6f}) = {f(x_new):.8f}")
            return x_new
        
        if abs(x_new) > 10:  # проверка на расходимость
            print("Метод расходится")
            return None
        
        x = x_new
    
    print("Превышено максимальное количество итераций")
    return x

def verification(roots):
    """Проверка найденных корней"""
    print("\n" + "=" * 60)
    print("ПРОВЕРКА РЕЗУЛЬТАТОВ")
    print("=" * 60)
    
    methods = ["Дихотомия", "Ньютон", "Итерация"]
    
    print(f"{'Метод':>12} {'Корень':>12} {'f(корень)':>15} {'|f(корень)|':>15}")
    print("-" * 60)
    
    for i, root in enumerate(roots):
        if root is not None:
            fx = f(root)
            print(f"{methods[i]:>12} {root:>12.6f} {fx:>15.8f} {abs(fx):>15.8f}")
        else:
            print(f"{methods[i]:>12} {'Не найден':>12} {'-':>15} {'-':>15}")
    
    # Сравнение корней между методами
    valid_roots = [r for r in roots if r is not None]
    if len(valid_roots) > 1:
        print(f"\nСравнение результатов:")
        for i in range(len(valid_roots)):
            for j in range(i+1, len(valid_roots)):
                diff = abs(valid_roots[i] - valid_roots[j])
                print(f"Разность между {methods[i]} и {methods[j]}: {diff:.8f}")

def main():
    """Главная функция"""
    print("РЕШЕНИЕ НЕЛИНЕЙНОГО УРАВНЕНИЯ: 3^x - 5x^2 + 1 = 0")
    print("Поиск наименьшего положительного корня")
    
    # Графическая локализация
    interval = plot_function()
    
    if interval is None:
        print("Не удалось найти положительный интервал с корнем")
        return
    
    a, b = interval
    tolerance = 0.001
    
    # Выбираем начальное приближение для итерационных методов
    x0 = (a + b) / 2
    
    # Решение всеми методами
    root_bisection = bisection_method(a, b, tolerance)
    root_newton = newton_method(x0, tolerance)
    root_iteration = iteration_method(x0, tolerance)
    
    # Проверка результатов
    roots = [root_bisection, root_newton, root_iteration]
    verification(roots)
    
    # Итоговый результат
    valid_roots = [r for r in roots if r is not None]
    if valid_roots:
        final_root = sum(valid_roots) / len(valid_roots)
        print(f"\n" + "=" * 60)
        print("ИТОГОВЫЙ РЕЗУЛЬТАТ")
        print("=" * 60)
        print(f"Наименьший положительный корень: {final_root:.6f}")
        print(f"Проверка: f({final_root:.6f}) = {f(final_root):.8f}")

if __name__ == "__main__":
    main()