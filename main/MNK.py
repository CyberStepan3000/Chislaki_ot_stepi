import numpy as np
import matplotlib.pyplot as plt

def calculate_sums(x, y):
    """Вычисление всех необходимых сумм для МНК"""
    sums = {
        'n': len(x),
        'sum_x': np.sum(x),
        'sum_x2': np.sum(x**2),
        'sum_x3': np.sum(x**3),
        'sum_x4': np.sum(x**4),
        'sum_y': np.sum(y),
        'sum_xy': np.sum(x*y),
        'sum_x2y': np.sum(x**2 * y)
    }
    return sums

def print_sums(sums):
    """Печать вычисленных сумм"""
    print("Вычисленные суммы:")
    print(f"n = {sums['n']}")
    print(f"∑xᵢ = {sums['sum_x']:.4f}")
    print(f"∑xᵢ² = {sums['sum_x2']:.4f}")
    print(f"∑xᵢ³ = {sums['sum_x3']:.4f}")
    print(f"∑xᵢ⁴ = {sums['sum_x4']:.4f}")
    print(f"∑yᵢ = {sums['sum_y']:.4f}")
    print(f"∑xᵢyᵢ = {sums['sum_xy']:.4f}")
    print(f"∑xᵢ²yᵢ = {sums['sum_x2y']:.4f}\n")

def linear_approximation(x, y, sums=None, verbose=True):
    """Линейная аппроксимация методом наименьших квадратов"""
    if sums is None:
        sums = calculate_sums(x, y)
    
    if verbose:
        print("\nЛинейная аппроксимация (1-я степень):")
        print("Нормальная система уравнений:")
        print(f"{sums['n']}·a0 + {sums['sum_x']:.1f}·a1 = {sums['sum_y']:.4f}")
        print(f"{sums['sum_x']:.1f}·a0 + {sums['sum_x2']:.1f}·a1 = {sums['sum_xy']:.4f}")
    
    # Матрица коэффициентов
    A = np.array([
        [sums['n'], sums['sum_x']],
        [sums['sum_x'], sums['sum_x2']]
    ])
    
    # Вектор правой части
    b = np.array([sums['sum_y'], sums['sum_xy']])
    
    # Решение системы
    coefficients = np.linalg.solve(A, b)
    
    # Вычисление ошибки
    y_pred = coefficients[0] + coefficients[1] * x
    error = np.sum((y - y_pred)**2)
    
    if verbose:
        print("\nМатрица системы:")
        print(A)
        print("Вектор правой части:")
        print(b)
        print(f"\nКоэффициенты: a0 = {coefficients[0]:.6f}, a1 = {coefficients[1]:.6f}")
        print(f"Сумма квадратов ошибок: {error:.6f}")
    
    return coefficients, error

def quadratic_approximation(x, y, sums=None, verbose=True):
    """Квадратичная аппроксимация методом наименьших квадратов"""
    if sums is None:
        sums = calculate_sums(x, y)
    
    if verbose:
        print("\nКвадратичная аппроксимация (2-я степень):")
        print("Нормальная система уравнений:")
        print(f"{sums['n']}·a0 + {sums['sum_x']:.1f}·a1 + {sums['sum_x2']:.1f}·a2 = {sums['sum_y']:.4f}")
        print(f"{sums['sum_x']:.1f}·a0 + {sums['sum_x2']:.1f}·a1 + {sums['sum_x3']:.1f}·a2 = {sums['sum_xy']:.4f}")
        print(f"{sums['sum_x2']:.1f}·a0 + {sums['sum_x3']:.1f}·a1 + {sums['sum_x4']:.1f}·a2 = {sums['sum_x2y']:.4f}")
    
    # Матрица коэффициентов
    A = np.array([
        [sums['n'], sums['sum_x'], sums['sum_x2']],
        [sums['sum_x'], sums['sum_x2'], sums['sum_x3']],
        [sums['sum_x2'], sums['sum_x3'], sums['sum_x4']]
    ])
    
    # Вектор правой части
    b = np.array([sums['sum_y'], sums['sum_xy'], sums['sum_x2y']])
    
    # Решение системы
    coefficients = np.linalg.solve(A, b)
    
    # Вычисление ошибки
    y_pred = coefficients[0] + coefficients[1] * x + coefficients[2] * x**2
    error = np.sum((y - y_pred)**2)
    
    if verbose:
        print("\nМатрица системы:")
        print(A)
        print("Вектор правой части:")
        print(b)
        print(f"\nКоэффициенты: a0 = {coefficients[0]:.6f}, a1 = {coefficients[1]:.6f}, a2 = {coefficients[2]:.6f}")
        print(f"Сумма квадратов ошибок: {error:.6f}")
    
    return coefficients, error

def plot_approximation(x, y, linear_coeffs, quad_coeffs, linear_error, quad_error):
    """Построение графиков аппроксимации"""
    x_plot = np.linspace(min(x)-0.5, max(x)+0.5, 100)
    
    # Вычисление значений для графиков
    y_linear = linear_coeffs[0] + linear_coeffs[1] * x_plot
    y_quad = quad_coeffs[0] + quad_coeffs[1] * x_plot + quad_coeffs[2] * x_plot**2
    
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color='red', label='Исходные точки', zorder=5)
    plt.plot(x_plot, y_linear, label=f'Линейная (ошибка={linear_error:.4f})')
    plt.plot(x_plot, y_quad, label=f'Квадратичная (ошибка={quad_error:.4f})')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Аппроксимация методом наименьших квадратов')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    # Исходные данные
    x = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0])
    y = np.array([-1.8415, 0.0, 1.8415, 2.9093, 3.1411, 3.2432])
    
    # Вычисление сумм один раз для обеих аппроксимаций
    sums = calculate_sums(x, y)
    print_sums(sums)
    
    # Линейная аппроксимация
    linear_coeffs, linear_error = linear_approximation(x, y, sums)
    
    # Квадратичная аппроксимация
    quad_coeffs, quad_error = quadratic_approximation(x, y, sums)
    
    # Построение графиков
    plot_approximation(x, y, linear_coeffs, quad_coeffs, linear_error, quad_error)

if __name__ == "__main__":
    main()