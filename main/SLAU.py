import numpy as np
import matplotlib.pyplot as plt

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
    
    def reorder_for_convergence(self):
        """Перестановка строк для достижения диагонального преобладания"""
        print("\n=== ПОПЫТКА ПЕРЕСТАНОВКИ СТРОК ===")
        
        # Находим оптимальную перестановку строк
        best_permutation = None
        best_score = -1
        
        from itertools import permutations
        
        for perm in permutations(range(self.n)):
            A_perm = self.A[list(perm), :]
            b_perm = self.b[list(perm)]
            
            # Проверяем диагональное преобладание
            score = 0
            for i in range(self.n):
                diagonal_elem = abs(A_perm[i, i])
                sum_other = sum(abs(A_perm[i, j]) for j in range(self.n) if j != i)
                if diagonal_elem > sum_other:
                    score += 1
            
            if score > best_score:
                best_score = score
                best_permutation = perm
                if score == self.n:  # Полное диагональное преобладание
                    break
        
        if best_permutation and best_score > 0:
            print(f"Найдена перестановка: {best_permutation}")
            print(f"Строк с диагональным преобладанием: {best_score}/{self.n}")
            
            self.A = self.A[list(best_permutation), :]
            self.b = self.b[list(best_permutation)]
            self.augmented_matrix = np.hstack([self.A.copy(), self.b.reshape(-1, 1)])
            
            print("\nНовая матрица A:")
            print(self.A)
            print("Новый вектор b:")
            print(self.b)
            
            return True
        else:
            print("Не удалось найти перестановку для диагонального преобладания")
            return False
    
    def simple_iteration_method(self, epsilon=0.01, max_iterations=1000):
        """Метод простых итераций (метод Якоби)"""
        print("\n=== МЕТОД ПРОСТЫХ ИТЕРАЦИЙ ===")
        
        # Преобразование к итерационному виду x = Bx + c
        D = np.diag(np.diag(self.A))
        L = np.tril(self.A, -1)
        U = np.triu(self.A, 1)
        
        # Проверка на нулевые диагональные элементы
        if np.any(np.diag(D) == 0):
            print("ОШИБКА: Обнаружен нулевой диагональный элемент!")
            return None, [], []
        
        B = -np.linalg.inv(D) @ (L + U)
        c = np.linalg.inv(D) @ self.b
        
        # Проверка спектрального радиуса
        rho = max(abs(np.linalg.eigvals(B)))
        print(f"Спектральный радиус итерационной матрицы: {rho:.6f}")
        
        if rho >= 1:
            print("ПРЕДУПРЕЖДЕНИЕ: Спектральный радиус >= 1, метод может не сходиться!")
        
        # Начальное приближение
        x = np.zeros(self.n)
        
        iterations = []
        errors = []
        
        print(f"{'Итерация':<10} {'x1':<12} {'x2':<12} {'x3':<12} {'Погрешность':<12}")
        print("-" * 60)
        
        for k in range(max_iterations):
            x_new = B @ x + c
            
            # Проверка на расходимость
            if np.any(np.isnan(x_new)) or np.any(np.isinf(x_new)) or np.linalg.norm(x_new) > 1e10:
                print(f"РАСХОДИМОСТЬ на итерации {k+1}!")
                return None, iterations, errors
            
            # Вычисление погрешности
            error = np.linalg.norm(x_new - x, ord=np.inf)
            
            iterations.append(k + 1)
            errors.append(error)
            
            print(f"{k+1:<10} {x_new[0]:<12.6f} {x_new[1]:<12.6f} {x_new[2]:<12.6f} {error:<12.6f}")
            
            if error < epsilon:
                print(f"\nСходимость достигнута за {k+1} итераций")
                return x_new, iterations, errors
            
            x = x_new.copy()
        
        print(f"\nМаксимальное количество итераций ({max_iterations}) достигнуто")
        return x, iterations, errors
    
    def relaxation_method(self, omega=0.5, epsilon=0.01, max_iterations=1000):
        """Метод релаксации (SOR - Successive Over-Relaxation)"""
        print(f"\n=== МЕТОД РЕЛАКСАЦИИ (ω = {omega}) ===")
        
        # Начальное приближение
        x = np.zeros(self.n)
        
        iterations = []
        errors = []
        
        print(f"{'Итерация':<10} {'x1':<12} {'x2':<12} {'x3':<12} {'Погрешность':<12}")
        print("-" * 60)
        
        for k in range(max_iterations):
            x_old = x.copy()
            
            # Обновление компонент вектора x с релаксацией
            for i in range(self.n):
                if abs(self.A[i, i]) < 1e-10:
                    print(f"ОШИБКА: Диагональный элемент A[{i},{i}] близок к нулю!")
                    return None, iterations, errors
                
                sum1 = sum(self.A[i, j] * x[j] for j in range(i))
                sum2 = sum(self.A[i, j] * x_old[j] for j in range(i + 1, self.n))
                
                # Классический шаг Зейделя
                x_seidel = (self.b[i] - sum1 - sum2) / self.A[i, i]
                
                # Релаксация
                x[i] = (1 - omega) * x_old[i] + omega * x_seidel
            
            # Проверка на расходимость
            if np.any(np.isnan(x)) or np.any(np.isinf(x)) or np.linalg.norm(x) > 1e10:
                print(f"РАСХОДИМОСТЬ на итерации {k+1}!")
                return None, iterations, errors
            
            # Вычисление погрешности
            error = np.linalg.norm(x - x_old, ord=np.inf)
            
            iterations.append(k + 1)
            errors.append(error)
            
            print(f"{k+1:<10} {x[0]:<12.6f} {x[1]:<12.6f} {x[2]:<12.6f} {error:<12.6f}")
            
            if error < epsilon:
                print(f"\nСходимость достигнута за {k+1} итераций")
                return x, iterations, errors
        
        print(f"\nМаксимальное количество итераций ({max_iterations}) достигнуто")
        return x, iterations, errors
    
    def jacobi_with_damping(self, alpha=0.5, epsilon=0.01, max_iterations=1000):
        """Метод Якоби с демпфированием"""
        print(f"\n=== МЕТОД ЯКОБИ С ДЕМПФИРОВАНИЕМ (α = {alpha}) ===")
        
        # Преобразование к итерационному виду
        D = np.diag(np.diag(self.A))
        L = np.tril(self.A, -1)
        U = np.triu(self.A, 1)
        
        if np.any(np.diag(D) == 0):
            print("ОШИБКА: Обнаружен нулевой диагональный элемент!")
            return None, [], []
        
        B = -np.linalg.inv(D) @ (L + U)
        c = np.linalg.inv(D) @ self.b
        
        # Начальное приближение
        x = np.zeros(self.n)
        
        iterations = []
        errors = []
        
        print(f"{'Итерация':<10} {'x1':<12} {'x2':<12} {'x3':<12} {'Погрешность':<12}")
        print("-" * 60)
        
        for k in range(max_iterations):
            x_old = x.copy()
            
            # Классический шаг Якоби
            x_jacobi = B @ x + c
            
            # Демпфирование
            x = (1 - alpha) * x_old + alpha * x_jacobi
            
            # Проверка на расходимость
            if np.any(np.isnan(x)) or np.any(np.isinf(x)) or np.linalg.norm(x) > 1e10:
                print(f"РАСХОДИМОСТЬ на итерации {k+1}!")
                return None, iterations, errors
            
            # Вычисление погрешности
            error = np.linalg.norm(x - x_old, ord=np.inf)
            
            iterations.append(k + 1)
            errors.append(error)
            
            print(f"{k+1:<10} {x[0]:<12.6f} {x[1]:<12.6f} {x[2]:<12.6f} {error:<12.6f}")
            
            if error < epsilon:
                print(f"\nСходимость достигнута за {k+1} итераций")
                return x, iterations, errors
        
        print(f"\nМаксимальное количество итераций ({max_iterations}) достигнуто")
        return x, iterations, errors
    
    def seidel_method(self, epsilon=0.01, max_iterations=1000):
        """Метод Зейделя"""
        print("\n=== МЕТОД ЗЕЙДЕЛЯ ===")
        
        # Начальное приближение
        x = np.zeros(self.n)
        
        iterations = []
        errors = []
        
        print(f"{'Итерация':<10} {'x1':<12} {'x2':<12} {'x3':<12} {'Погрешность':<12}")
        print("-" * 60)
        
        for k in range(max_iterations):
            x_old = x.copy()
            
            # Обновление компонент вектора x
            for i in range(self.n):
                if abs(self.A[i, i]) < 1e-10:
                    print(f"ОШИБКА: Диагональный элемент A[{i},{i}] близок к нулю!")
                    return None, iterations, errors
                
                sum1 = sum(self.A[i, j] * x[j] for j in range(i))
                sum2 = sum(self.A[i, j] * x_old[j] for j in range(i + 1, self.n))
                x[i] = (self.b[i] - sum1 - sum2) / self.A[i, i]
            
            # Проверка на расходимость
            if np.any(np.isnan(x)) or np.any(np.isinf(x)) or np.linalg.norm(x) > 1e10:
                print(f"РАСХОДИМОСТЬ на итерации {k+1}!")
                return None, iterations, errors
            
            # Вычисление погрешности
            error = np.linalg.norm(x - x_old, ord=np.inf)
            
            iterations.append(k + 1)
            errors.append(error)
            
            print(f"{k+1:<10} {x[0]:<12.6f} {x[1]:<12.6f} {x[2]:<12.6f} {error:<12.6f}")
            
            if error < epsilon:
                print(f"\nСходимость достигнута за {k+1} итераций")
                return x, iterations, errors
        
        print(f"\nМаксимальное количество итераций ({max_iterations}) достигнуто")
        return x, iterations, errors
    
    def check_solution(self, x):
        """Проверка решения подстановкой в исходную систему"""
        print("\n=== ПРОВЕРКА РЕШЕНИЯ ===")
        residual = self.A @ x - self.b
        print(f"Решение: x1 = {x[0]:.6f}, x2 = {x[1]:.6f}, x3 = {x[2]:.6f}")
        print(f"Невязка: {residual}")
        print(f"Норма невязки: {np.linalg.norm(residual):.8f}")
        
        # Подстановка в каждое уравнение
        print("\nПроверка каждого уравнения:")
        for i in range(self.n):
            left_side = sum(self.A[i, j] * x[j] for j in range(self.n))
            print(f"Уравнение {i+1}: {left_side:.6f} = {self.b[i]:.6f} (разность: {abs(left_side - self.b[i]):.8f})")
    
    def plot_convergence_multiple(self, methods_data):
        """Построение графика сходимости для нескольких методов"""
        plt.figure(figsize=(12, 8))
        colors = ['b-o', 'r-s', 'g-^', 'm-d', 'c-*']
        
        for i, (name, iterations, errors) in enumerate(methods_data):
            if len(iterations) > 0 and len(errors) > 0:
                plt.semilogy(iterations, errors, colors[i % len(colors)], 
                           label=name, markersize=4, linewidth=2)
        
        plt.xlabel('Номер итерации')
        plt.ylabel('Погрешность (логарифмическая шкала)')
        plt.title('Сравнение методов решения СЛАУ')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.show()

    