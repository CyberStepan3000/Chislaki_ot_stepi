import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations

class Slau:
    def __init__(self, A: np.ndarray, b: np.ndarray):
        self.A = A.astype(float)
        self.b = b.astype(float)
        self.n = A.shape[0]
        self.augmented_matrix = np.hstack([self.A.copy(), self.b.reshape(-1, 1)])

    def gauss_jordan(self, pivoting: bool = True):
        """
        Метод Гаусса-Жордана для решения системы линейных уравнений Ax = b.
        

        """
        mat = self.augmented_matrix.copy()
        n = self.n

        for i in range(n):
            if pivoting:
                max_row = np.argmax(abs(mat[i:, i])) + i
                if i != max_row:
                    mat[[i, max_row]] = mat[[max_row, i]]
            
            pivot = mat[i, i]
            if np.isclose(pivot, 0):
                raise ValueError("Матрица вырождена — нет уникального решения.")

            mat[i] = mat[i] / pivot

            for j in range(n):
                if i != j:
                    mat[j] -= mat[i] * mat[j, i]

        solution = mat[:, -1]
        return solution, self._inverse_matrix(), self._determinant(), self._verify_solution(solution), self._verify_inverse(self._inverse_matrix())

    def _determinant(self):
        print("Determinant =", np.linalg.det(self.A))
        return round(np.linalg.det(self.A), 6)

    def _inverse_matrix(self):
        print("Inverse matrix = \n", np.linalg.inv(self.A))
        return np.linalg.inv(self.A)

    def _verify_solution(self, x: np.ndarray):
        Ax = self.A @ x
        print('verification = \n', Ax) 
        if np.allclose(Ax, self.b):
            print("Verification successful")
            return True
        else:
            print("Verification failed")
            return False

    def _verify_inverse(self, A_inv: np.ndarray):
        identity = np.eye(self.n)
        print('A @ A_inv = \n', self.A @ A_inv)
        print('identity = \n', identity)
        print('verification = \n', np.allclose(self.A @ A_inv, identity))
        return np.allclose(self.A @ A_inv, identity)
    
    def progonka(self):
        """
        Метод прогонки для трёхдиагональных матриц.
        Решает систему Ax = b, где A - трёхдиагональная матрица.    
        """
        n = self.n
        A = self.A
        d = self.b

        # Проверка: является ли матрица трёхдиагональной
        if not np.allclose(A, np.triu(np.tril(A, 1), -1)):
            raise ValueError("Матрица не трёхдиагональная — метод прогонки неприменим.")

        # Инициализация диагоналей
        # a — поддиагональ, b — главная, c — наддиагональ
        a = np.zeros(n)
        b = np.zeros(n)
        c = np.zeros(n)

        for i in range(n):
            b[i] = A[i, i]                  # главная диагональ
            if i > 0:
                a[i] = A[i, i - 1]          # поддиагональ (слева от главной)
            if i < n - 1:
                c[i] = A[i, i + 1]          # наддиагональ (справа от главной)
        
        # Прямой ход
        cp = np.zeros(n)
        dp = np.zeros(n)
        cp[0] = c[0] / b[0]
        dp[0] = d[0] / b[0]

        for i in range(1, n):
                denom = b[i] - a[i] * cp[i - 1]  # знаменатель 
                if np.isclose(denom, 0):
                    raise ZeroDivisionError(f"Нулевой знаменатель в строке {i}, решение невозможно.")
                cp[i] = c[i] / denom if i < n - 1 else 0  # последняя c не используется
                dp[i] = (d[i] - a[i] * dp[i - 1]) / denom
        # Обратный ход 
        x = np.zeros(n)
        x[-1] = dp[-1]  # последний элемент
        
        for i in reversed(range(n - 1)):
            x[i] = dp[i] - cp[i] * x[i + 1]

        
        return x



    def check_convergence(self):
        """Проверка условий сходимости для итерационных методов"""
        # Проверка диагонального преобладания
        diagonal_dominant = True
        for i in range(self.n):
            diagonal_elem = abs(self.A[i, i])
            sum_other = sum(abs(self.A[i, j]) for j in range(self.n) if j != i)
            if diagonal_elem <= sum_other:
                diagonal_dominant = False
                break
        
        print(f"Диагональное преобладание: {'Да' if diagonal_dominant else 'Нет'}")
        
        # Вычисление спектрального радиуса итерационной матрицы
        D = np.diag(np.diag(self.A))
        L = np.tril(self.A, -1)
        U = np.triu(self.A, 1)
        
        # Матрица итераций для метода простых итераций
        B_jacobi = -np.linalg.inv(D) @ (L + U)
        rho_jacobi = max(abs(np.linalg.eigvals(B_jacobi)))
        
        print(f"Спектральный радиус для метода Якоби: {rho_jacobi:.6f}")
        print(f"Сходимость метода Якоби: {'Да' if rho_jacobi < 1 else 'Нет'}")
        
        return diagonal_dominant, rho_jacobi < 1
    
    
    def __init__(self, A: np.ndarray, b: np.ndarray):
        self.A = A.astype(float)
        self.b = b.astype(float)
        self.n = A.shape[0]
        self.original_A = A.copy()
        self.original_b = b.copy()

    def make_diagonally_dominant(self):
        """Пытается переставить строки матрицы, чтобы сделать её диагонально преобладающей"""
        for perm in permutations(range(self.n)):
            A_perm = self.A[list(perm), :]
            b_perm = self.b[list(perm)]
            if all(abs(A_perm[i, i]) >= sum(abs(A_perm[i, j]) for j in range(self.n) if j != i) for i in range(self.n)):
                self.A = A_perm
                self.b = b_perm
                print("Матрица успешно преобразована к диагонально преобладающему виду")
                return True
        print("Не удалось привести матрицу к диагонально преобладающему виду")
        return False

    def restore_original(self):
        self.A = self.original_A.copy()
        self.b = self.original_b.copy()
        print("Матрица и вектор b восстановлены до исходного состояния")

    def simple_iteration_method(self, epsilon=0.01, max_iterations=100): 
        print("\n=== МЕТОД ПРОСТЫХ ИТЕРАЦИЙ (ЯКОБИ) ===")
        D = np.diag(np.diag(self.A))
        L = np.tril(self.A, -1)
        U = np.triu(self.A, 1)
        
        if np.any(np.diag(D) == 0):
            print("ОШИБКА: Обнаружен нулевой диагональный элемент!")
            return None, [], []

        B = -np.linalg.inv(D) @ (L + U)
        c = np.linalg.inv(D) @ self.b

        rho = max(abs(np.linalg.eigvals(B)))
        print(f"Спектральный радиус: {rho:.6f}")

        if rho >= 1:
            print("ПРЕДУПРЕЖДЕНИЕ: Метод может не сходиться (rho >= 1)")

        x = np.zeros(self.n)
        iterations, errors = [], []

        print(f"{'Итерация':<10} {'x':<36} {'Погрешность':<12}")
        print("-" * 60)

        for k in range(max_iterations):
            x_new = B @ x + c
            error = np.linalg.norm(x_new - x, ord=np.inf)
            iterations.append(k + 1)
            errors.append(error)
            print(f"{k+1:<10} {str(np.round(x_new, 6)):<36} {error:<12.6f}")
            if error < epsilon:
                print(f"\nСходимость достигнута за {k+1} итераций")
                return x_new, iterations, errors
            x = x_new.copy()

        print("\nМаксимальное число итераций достигнуто")
        return x, iterations, errors

    def seidel_method(self, epsilon=0.01, max_iterations=100):
        print("\n=== МЕТОД ЗЕЙДЕЛЯ ===")
        x = np.zeros(self.n)
        iterations, errors = [], []

        print(f"{'Итерация':<10} {'x':<36} {'Погрешность':<12}")
        print("-" * 60)

        for k in range(max_iterations):
            x_old = x.copy()
            for i in range(self.n):
                sum1 = sum(self.A[i, j] * x[j] for j in range(i))
                sum2 = sum(self.A[i, j] * x_old[j] for j in range(i + 1, self.n))
                if abs(self.A[i, i]) < 1e-10:
                    print(f"ОШИБКА: A[{i},{i}] близок к нулю!")
                    return None, iterations, errors
                x[i] = (self.b[i] - sum1 - sum2) / self.A[i, i]
            error = np.linalg.norm(x - x_old, ord=np.inf)
            iterations.append(k + 1)
            errors.append(error)
            print(f"{k+1:<10} {str(np.round(x, 6)):<36} {error:<12.6f}")
            if error < epsilon:
                print(f"\nСходимость достигнута за {k+1} итераций")
                return x, iterations, errors

        print("\nМаксимальное число итераций достигнуто")
        return x, iterations, errors

    def check_solution(self, x):
        print("\n=== ПРОВЕРКА РЕШЕНИЯ ===")
        residual = self.A @ x - self.b
        print(f"Решение: {np.round(x, 6)}")
        print(f"Невязка: {residual}")
        print(f"Норма невязки: {np.linalg.norm(residual):.8f}")
        for i in range(self.n):
            left = sum(self.A[i, j] * x[j] for j in range(self.n))
            print(f"Уравнение {i+1}: {left:.6f} = {self.b[i]:.6f} (Δ={abs(left - self.b[i]):.8f})")

def jacobi_eigen_with_check(A, epsilon=0.0001, check_tolerance=1e-5):
    """
    Находит собственные значения и векторы методом Якоби 
    и проверяет A * v = λ * v с использованием исходной матрицы.

    Параметры:
        A - исходная симметричная матрица (numpy array)
        epsilon - точность для метода Якоби (float)
        check_tolerance - допустимая погрешность проверки (float)

    Возвращает:
        eigenvalues - собственные значения
        eigenvectors - собственные векторы (по столбцам)
    """
    n = A.shape[0]
    A_current = A.copy()  # Рабочая копия матрицы
    eigenvectors = np.eye(n)
    
    # --- Метод Якоби ---
    while True:
        # Находим максимальный недиагональный элемент
        max_val = 0
        p, q = 0, 0
        for i in range(n):
            for j in range(i + 1, n):
                if abs(A_current[i, j]) > max_val:
                    max_val = abs(A_current[i, j])
                    p, q = i, j
        
        if max_val < epsilon:
            break
        
        # Вычисляем угол поворота
        if np.isclose(A_current[p, p], A_current[q, q]):
            theta = np.pi / 4
        else:
            theta = 0.5 * np.arctan(2 * A_current[p, q] / (A_current[p, p] - A_current[q, q]))
        
        # Матрица вращения
        rotation = np.eye(n)
        c = np.cos(theta)
        s = np.sin(theta)
        rotation[p, p] = c
        rotation[q, q] = c
        rotation[p, q] = -s
        rotation[q, p] = s
        
        # Применяем вращение
        A_current = rotation.T @ A_current @ rotation
        eigenvectors = eigenvectors @ rotation
    
    eigenvalues = np.diag(A_current)

    # --- Проверка с ИСХОДНОЙ матрицей A ---
    print("\n🔹 Проверка A * v = λ * v:")
    all_ok = True
    for i in range(n):
        λ = eigenvalues[i]
        v = eigenvectors[:, i]
        
        Av = A @ v  # Умножаем на исходную матрицу!
        λv = λ * v
        
        error = np.linalg.norm(Av - λv)
        print(f"λ_{i} = {λ:.6f}: Ошибка = {error:.10f}", end=" ")
        
        if error < check_tolerance:
            print(" (OK)")  
        else:
            print("(not OK) (Ошибка слишком велика!)")
            all_ok = False
    
    if all_ok:
        print("\nВсе проверки пройдены успешно!")
    else:
        print("\nВнимание: есть ошибки в вычислениях!")
    
    return eigenvalues, eigenvectors
