import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# Система уравнений:
# x1 - cos(x2) = a
# x2 - sin(x1) = a

# Значение параметра a
a = 0.5

def f1(x1, x2):
    """Первое уравнение: x1 - cos(x2) - a = 0"""
    return x1 - np.cos(x2) - a

def f2(x1, x2):
    """Второе уравнение: x2 - sin(x1) - a = 0"""
    return x2 - np.sin(x1) - a

def system(vars):
    """Система уравнений в векторном виде"""
    x1, x2 = vars
    return [f1(x1, x2), f2(x1, x2)]

def jacobian(x1, x2):
    """Матрица Якоби системы"""
    J = np.array([
        [1, np.sin(x2)],
        [-np.cos(x1), 1]
    ])
    return J

def newton_method(x0, tol=0.001, max_iter=100):
    """Метод Ньютона для решения системы"""
    x = np.array(x0)
    iterations = []
    
    print("Метод Ньютона:")
    print(f"Начальное приближение: x1 = {x[0]:.6f}, x2 = {x[1]:.6f}")
    
    for i in range(max_iter):
        f_val = np.array([f1(x[0], x[1]), f2(x[0], x[1])])
        J = jacobian(x[0], x[1])
        
        # Проверка на вырожденность матрицы Якоби
        det_J = np.linalg.det(J)
        if abs(det_J) < 1e-12:
            print("Матрица Якоби вырождена!")
            break
            
        # Решение системы J * dx = -f
        try:
            dx = np.linalg.solve(J, -f_val)
        except np.linalg.LinAlgError:
            print("Не удается решить систему для матрицы Якоби")
            break
            
        x_new = x + dx
        iterations.append(x_new.copy())
        
        # Проверка сходимости
        if np.linalg.norm(dx) < tol:
            print(f"Сходимость достигнута за {i+1} итераций")
            print(f"Решение: x1 = {x_new[0]:.6f}, x2 = {x_new[1]:.6f}")
            return x_new, iterations
            
        x = x_new
        print(f"Итерация {i+1}: x1 = {x[0]:.6f}, x2 = {x[1]:.6f}, ||dx|| = {np.linalg.norm(dx):.6f}")
    
    print(f"Максимальное количество итераций ({max_iter}) достигнуто")
    return x, iterations

def iteration_method(x0, tol=0.001, max_iter=100):
    """Метод простых итераций"""
    # Преобразуем систему к виду:
    # x1 = cos(x2) + a
    # x2 = sin(x1) + a
    
    def phi1(x1, x2):
        return np.cos(x2) + a
    
    def phi2(x1, x2):
        return np.sin(x1) + a
    
    x = np.array(x0)
    iterations = []
    
    print("\nМетод простых итераций:")
    print(f"Начальное приближение: x1 = {x[0]:.6f}, x2 = {x[1]:.6f}")
    
    for i in range(max_iter):
        x_new = np.array([phi1(x[0], x[1]), phi2(x[0], x[1])])
        iterations.append(x_new.copy())
        
        # Проверка сходимости
        if np.linalg.norm(x_new - x) < tol:
            print(f"Сходимость достигнута за {i+1} итераций")
            print(f"Решение: x1 = {x_new[0]:.6f}, x2 = {x_new[1]:.6f}")
            return x_new, iterations
            
        x = x_new
        print(f"Итерация {i+1}: x1 = {x[0]:.6f}, x2 = {x[1]:.6f}")
    
    print(f"Максимальное количество итераций ({max_iter}) достигнuto")
    return x, iterations

def check_solution(x):
    """Проверка решения"""
    print(f"\nПроверка решения:")
    print(f"x1 = {x[0]:.6f}, x2 = {x[1]:.6f}")
    
    f1_val = f1(x[0], x[1])
    f2_val = f2(x[0], x[1])
    
    print(f"f1(x1, x2) = x1 - cos(x2) - a = {f1_val:.6f}")
    print(f"f2(x1, x2) = x2 - sin(x1) - a = {f2_val:.6f}")
    print(f"Норма невязки: {np.sqrt(f1_val**2 + f2_val**2):.6f}")

def plot_system():
    """Графическая визуализация системы"""
    # Создание сетки точек
    x1 = np.linspace(-2, 4, 400)
    x2 = np.linspace(-2, 4, 400)
    X1, X2 = np.meshgrid(x1, x2)
    
    # Вычисление значений функций
    F1 = X1 - np.cos(X2) - a
    F2 = X2 - np.sin(X1) - a
    
    plt.figure(figsize=(12, 10))
    
    # График линий уровня
    plt.subplot(2, 2, 1)
    plt.contour(X1, X2, F1, levels=[0], colors='red', linewidths=2,)  # label='f1 = 0')
    plt.contour(X1, X2, F2, levels=[0], colors='blue', linewidths=2,) # label='f2 = 0')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title(f'Графическая локализация корней (a = {a})')
    plt.grid(True, alpha=0.3)
    plt.legend(['x1 - cos(x2) = a', 'x2 - sin(x1) = a'])
    
    # Поиск приблизительных корней для визуализации
    roots = []
    for x1_start in np.arange(-1, 3, 0.5):
        for x2_start in np.arange(-1, 3, 0.5):
            try:
                root = fsolve(system, [x1_start, x2_start])
                # Проверка, что это действительно корень
                if abs(f1(root[0], root[1])) < 0.01 and abs(f2(root[0], root[1])) < 0.01:
                    # Проверка, что корень не дубликат
                    is_new = True
                    for existing_root in roots:
                        if np.linalg.norm(root - existing_root) < 0.1:
                            is_new = False
                            break
                    if is_new:
                        roots.append(root)
            except:
                pass
    
    # Отображение найденных корней
    for i, root in enumerate(roots):
        plt.plot(root[0], root[1], 'go', markersize=8, label=f'Корень {i+1}' if i == 0 else "")
        plt.annotate(f'({root[0]:.2f}, {root[1]:.2f})', 
                    (root[0], root[1]), xytext=(5, 5), textcoords='offset points')
    
    if roots:
        plt.legend()
    
    # 3D визуализация
    plt.subplot(2, 2, 2)
    x1_3d = np.linspace(-1, 3, 50)
    x2_3d = np.linspace(-1, 3, 50)
    X1_3d, X2_3d = np.meshgrid(x1_3d, x2_3d)
    F1_3d = X1_3d - np.cos(X2_3d) - a
    
    ax = plt.gca()
    contour = ax.contourf(X1_3d, X2_3d, F1_3d, levels=20, cmap='RdYlBu')
    plt.colorbar(contour)
    ax.contour(X1_3d, X2_3d, F1_3d, levels=[0], colors='black', linewidths=2)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('f1(x1, x2) = x1 - cos(x2) - a')
    
    plt.subplot(2, 2, 3)
    F2_3d = X2_3d - np.sin(X1_3d) - a
    contour = plt.contourf(X1_3d, X2_3d, F2_3d, levels=20, cmap='RdYlBu')
    plt.colorbar(contour)
    plt.contour(X1_3d, X2_3d, F2_3d, levels=[0], colors='black', linewidths=2)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('f2(x1, x2) = x2 - sin(x1) - a')
    
    # График сходимости
    plt.subplot(2, 2, 4)
    plt.text(0.1, 0.5, f'Система уравнений:\nx1 - cos(x2) = {a}\nx2 - sin(x1) = {a}\n\nПоложительные корни будут\nнайдены численными методами', 
             transform=plt.gca().transAxes, fontsize=12, verticalalignment='center')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return roots

# Основная программа
print("Решение системы нелинейных уравнений:")
print(f"x1 - cos(x2) = {a}")
print(f"x2 - sin(x1) = {a}")
print("="*50)

# Графическая визуализация
print("Графическая локализация корней...")
roots = plot_system()

# Выбор положительного начального приближения
if roots:
    positive_roots = [root for root in roots if root[0] > 0 and root[1] > 0]
    if positive_roots:
        x0 = positive_roots[0]
        print(f"\nИспользуем положительное начальное приближение: x1 = {x0[0]:.3f}, x2 = {x0[1]:.3f}")
    else:
        x0 = [1.0, 1.0]  # Значение по умолчанию
        print(f"\nПоложительные корни не найдены автоматически. Используем x0 = [1.0, 1.0]")
else:
    x0 = [1.0, 1.0]  # Значение по умолчанию
    print(f"\nИспользуем начальное приближение по умолчанию: x0 = [1.0, 1.0]")

print("="*50)

# Решение методом Ньютона
try:
    newton_solution, newton_iterations = newton_method(x0)
    check_solution(newton_solution)
except Exception as e:
    print(f"Ошибка в методе Ньютона: {e}")

print("="*50)

# Решение методом итераций
try:
    iteration_solution, iteration_iterations = iteration_method(x0)
    check_solution(iteration_solution)
except Exception as e:
    print(f"Ошибка в методе итераций: {e}")

print("="*50)

# Сравнение методов
if 'newton_solution' in locals() and 'iteration_solution' in locals():
    print("Сравнение методов:")
    print(f"Метод Ньютона: x1 = {newton_solution[0]:.6f}, x2 = {newton_solution[1]:.6f}")
    print(f"Метод итераций: x1 = {iteration_solution[0]:.6f}, x2 = {iteration_solution[1]:.6f}")
    print(f"Разность решений: ||x_Newton - x_Iteration|| = {np.linalg.norm(newton_solution - iteration_solution):.6f}")