from SLAU import *

### ----- Первая задача -----

# A = np.array([
#     [1, 1, 20],
#     [215, 20, 1],
#     [20, 1, 1]
# ])
# b = np.array([12, 215, 2])

# solver = Slau(A, b)

#print("A = \n", A)
#x = solver.gauss_jordan()
#print (x)
### ----- Вторая задача -----

# A = np.array([
#     [-11,  9,   0,   0,  0],
#     [ 1,  -8,  1,   0,  0],
#     [ 0,  -2, -11,  5,  0],
#     [ 0,   0,  3, -14,  7],
#     [ 0,   0,  0,   8, 10]
# ], dtype=float)

# b = np.array([-114, 81, -8, -38, 144], dtype=float)

# solver = Slau(A, b)
# print("A = \n", A)
# x = solver.progonka()
# print (f'x = \n', x)
# solver._verify_solution(x)


### ----- Третья и четвертая задача + сравнивание методов -----

A = np.array([
    [1, 1, 20],
    [215, 20, 1],
    [20, 1, 1]
], dtype=float)

b = np.array([12, 215, 2], dtype=float)

print("Исходная система уравнений:")
print("x1 + x2 + 20x3 = 12")
print("215x1 + 20x2 + x3 = 215")
print("20x1 + x2 + x3 = 2")
print("\nМатрица A:")
print(A)
print("\nВектор b:")
print(b)

system = Slau(A, b)

diagonal_dom, convergent = system.check_convergence()

# Если нет сходимости, пытаемся переставить строки
if not diagonal_dom:
    print("\n" + "="*50)
    print("ПРОБЛЕМА: Матрица не имеет диагонального преобладания!")
    print("Это может привести к расходимости итерационных методов.")
    print("="*50)
    
    # Попытка перестановки строк
    reordered = system.reorder_for_convergence()
    
    if reordered:
        print("\nПроверка сходимости после перестановки:")
        system.check_convergence()
    else:
        print("\nПРЕДУПРЕЖДЕНИЕ: Система может не сходиться!")
        print("Будем использовать релаксационный параметр...")

# Точное решение для сравнения
print("\n=== ТОЧНОЕ РЕШЕНИЕ (методом Гаусса) ===")
exact_solution = np.linalg.solve(A, b)
print(f"Точное решение: x1 = {exact_solution[0]:.6f}, x2 = {exact_solution[1]:.6f}, x3 = {exact_solution[2]:.6f}")

# Список для хранения результатов всех методов
methods_results = []
methods_data = []

# Решение методом простых итераций
print("\n" + "="*60)
solution1, iter1, err1 = system.simple_iteration_method(epsilon=0.01, max_iterations=30)
if solution1 is not None:
    methods_results.append(('Простые итерации', solution1))
    methods_data.append(('Простые итерации', iter1, err1))
    system.check_solution(solution1)

# Решение методом Зейделя
print("\n" + "="*60)
solution2, iter2, err2 = system.seidel_method(epsilon=0.01, max_iterations=30)
if solution2 is not None:
    methods_results.append(('Метод Зейделя', solution2))
    methods_data.append(('Метод Зейделя', iter2, err2))
    system.check_solution(solution2)

# Пробуем разные параметры релаксации
print("\n" + "="*60)
print("ПРОБУЕМ МЕТОДЫ С РЕЛАКСАЦИЕЙ...")

omega_values = [0.1, 0.3, 0.5, 0.7, 0.9]
best_relaxation = None
best_relaxation_iter = float('inf')

for omega in omega_values:
    print(f"\n--- Тестируем ω = {omega} ---")
    solution_rel, iter_rel, err_rel = system.relaxation_method(omega=omega, epsilon=0.01, max_iterations=100)
    if solution_rel is not None:
        methods_results.append((f'Релаксация ω={omega}', solution_rel))
        methods_data.append((f'Релаксация ω={omega}', iter_rel, err_rel))
        if len(iter_rel) < best_relaxation_iter:
            best_relaxation = (omega, solution_rel, iter_rel, err_rel)
            best_relaxation_iter = len(iter_rel)
        print(f"УСПЕХ! Сошелся за {len(iter_rel)} итераций")
    else:
        print("Не сошелся")

# Пробуем демпфированный Якоби
print("\n" + "="*60)
print("ПРОБУЕМ ДЕМПФИРОВАННЫЙ ЯКОБИ...")

alpha_values = [0.1, 0.3, 0.5, 0.7]
best_jacobi = None
best_jacobi_iter = float('inf')

for alpha in alpha_values:
    print(f"\n--- Тестируем α = {alpha} ---")
    solution_jac, iter_jac, err_jac = system.jacobi_with_damping(alpha=alpha, epsilon=0.01, max_iterations=100)
    if solution_jac is not None:
        methods_results.append((f'Якоби α={alpha}', solution_jac))
        methods_data.append((f'Якоби α={alpha}', iter_jac, err_jac))
        if len(iter_jac) < best_jacobi_iter:
            best_jacobi = (alpha, solution_jac, iter_jac, err_jac)
            best_jacobi_iter = len(iter_jac)
        print(f"УСПЕХ! Сошелся за {len(iter_jac)} итераций")
    else:
        print("Не сошелся")

# Итоговые результаты
print("\n" + "="*80)
print("=== ИТОГОВЫЕ РЕЗУЛЬТАТЫ ===")
print("="*80)

if len(methods_results) > 0:
    print(f"{'Метод':<25} {'x1':<12} {'x2':<12} {'x3':<12} {'Погрешность':<12}")
    print("-" * 80)
    
    for method_name, solution in methods_results:
        error = np.linalg.norm(solution - exact_solution)
        print(f"{method_name:<25} {solution[0]:<12.6f} {solution[1]:<12.6f} {solution[2]:<12.6f} {error:<12.8f}")
    
    print(f"{'Точное решение':<25} {exact_solution[0]:<12.6f} {exact_solution[1]:<12.6f} {exact_solution[2]:<12.6f} {'0.00000000':<12}")
    
    # Лучшие методы
    if best_relaxation:
        print(f"\nЛучший метод релаксации: ω = {best_relaxation[0]}, {len(best_relaxation[2])} итераций")
    
    if best_jacobi:
        print(f"Лучший демпфированный Якоби: α = {best_jacobi[0]}, {len(best_jacobi[2])} итераций")
    
    # График сходимости
    if len(methods_data) > 0:
        system.plot_convergence_multiple(methods_data)
        
else:
    print("НИ ОДИН ИТЕРАЦИОННЫЙ МЕТОД НЕ СОШЕЛСЯ!")
    print("\nВозможные причины:")
    print("1. Матрица плохо обусловлена")
    print("2. Спектральный радиус слишком велик")
    print("3. Нужны более продвинутые методы предобуславливания")
    
    print(f"\nТочное решение: x1 = {exact_solution[0]:.6f}, x2 = {exact_solution[1]:.6f}, x3 = {exact_solution[2]:.6f}")