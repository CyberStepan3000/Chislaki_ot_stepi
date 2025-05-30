from SLAU import *

### ----- Первая задача -----

# A = np.array([
#     [1, 1, 20],
#     [215, 20, 1],
#     [20, 1, 1]
# ])
# b = np.array([12, 215, 2])

# solver = Slau(A, b)

# print("A = \n", A)
# x = solver.gauss_jordan()
# print (x)

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
    [1.0, 0.093, 0.00465],
    [0.05, 0.05, 1.0],
    [1.0, 0.05, 0.05]
])

b = np.array([1.0, 0.6, 0.1])

print("Исходная система уравнений:")
print("x1 + x2 + 20x3 = 12")
print("215x1 + 20x2 + x3 = 215")
print("20x1 + x2 + x3 = 2")
print("\nМатрица A:")
print(A)
print("\nВектор b:")
print(b)


slau = Slau(A, b)
print("Исходная матрица A и вектор b:")
print(slau.A)
print(slau.b)

# Попробуем сделать матрицу диагонально преобладающей
slau.make_diagonally_dominant()

# Метод простых итераций (Якоби)
x_jacobi, iter_jacobi, errors_jacobi = slau.simple_iteration_method(epsilon=0.001)
if x_jacobi is not None:
    slau.check_solution(x_jacobi)

# Восстановим исходную систему
slau.restore_original()

# Метод Зейделя
x_seidel, iter_seidel, errors_seidel = slau.seidel_method(epsilon=0.001)
if x_seidel is not None:
    slau.check_solution(x_seidel)

# Выводим информацию об итерациях для обоих методов
print("\nДетали итераций (Якоби):")
for i, err in zip(iter_jacobi, errors_jacobi):
    print(f"Итерация {i}: погрешность = {err:.6f}")

print("\nДетали итераций (Зейдель):")
for i, err in zip(iter_seidel, errors_seidel):
    print(f"Итерация {i}: погрешность = {err:.6f}")
    
### ----- Пятая задача -----

# A = np.array([
#     [1, 22, 1],
#     [22, 1, 1],
#     [1, 1, 23]
# ], dtype=float)

# eigenvalues, eigenvectors = jacobi_eigen_with_check(A, 0.01)

# print("Собственные значения:")
# print(eigenvalues)
# print("\nСобственные векторы (по столбцам):")
# print(eigenvectors)
