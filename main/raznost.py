import numpy as np


def solve_bvp(h):
    a_val = 1.0
    b_val = 3.0
    N = int(round((b_val - a_val) / h))
    x = [a_val + i * h for i in range(N + 1)]
    y0 = 3.0  # Известное значение в x0 = 1

    # Инициализация массивов для метода прогонки
    n_unknown = N  # Неизвестные: y1, y2, ..., yN
    a_coeff = np.zeros(n_unknown)
    b_coeff = np.zeros(n_unknown)
    c_coeff = np.zeros(n_unknown)
    d_coeff = np.zeros(n_unknown)

    # Уравнение для i=1 (соответствует узлу x1)
    xi = x[1]
    A1 = xi * (xi - 1) + (xi * h / 2)
    B1 = -2 * xi * (xi - 1) + h**2
    C1 = xi * (xi - 1) - (xi * h / 2)
    a_coeff[0] = 0.0  # Нет элемента y_{i-1} для первого уравнения
    b_coeff[0] = B1
    c_coeff[0] = C1
    d_coeff[0] = -A1 * y0

    # Уравнения для внутренних узлов (i=2 до N-1)
    for idx in range(1, n_unknown - 1):
        xi = x[idx + 1]  # Узел x_{i} = x_{idx+1}
        A_i = xi * (xi - 1) + (xi * h / 2)
        B_i = -2 * xi * (xi - 1) + h**2
        C_i = xi * (xi - 1) - (xi * h / 2)
        a_coeff[idx] = A_i
        b_coeff[idx] = B_i
        c_coeff[idx] = C_i
        d_coeff[idx] = 0.0

    # Уравнение для последнего узла (i=N, xN=3)
    if n_unknown >= 1:  # Если есть хотя бы одно неизвестное
        xN = x[-1]
        D_N = (2 * xN * (xN - 1)) / h**2
        term = (2 * xN * (xN - 1)) / h**2
        E_N = term * (-1 + h / 3) - xN / 3 + 1
        F_N = - (8 * xN * (xN - 1)) / (3 * h) + (4 * xN) / 3
        a_coeff[-1] = D_N
        b_coeff[-1] = E_N
        c_coeff[-1] = 0.0
        d_coeff[-1] = F_N

    # Метод прогонки
    n = n_unknown
    alpha = np.zeros(n)
    beta = np.zeros(n)

    # Прямой ход
    alpha[0] = -c_coeff[0] / b_coeff[0]
    beta[0] = d_coeff[0] / b_coeff[0]
    for i in range(1, n):
        denom = b_coeff[i] + a_coeff[i] * alpha[i - 1]
        alpha[i] = -c_coeff[i] / denom
        beta[i] = (d_coeff[i] - a_coeff[i] * beta[i - 1]) / denom

    # Обратный ход
    y_unknown = np.zeros(n)
    y_unknown[-1] = beta[-1]
    for i in range(n - 2, -1, -1):
        y_unknown[i] = alpha[i] * y_unknown[i + 1] + beta[i]

    # Сбор полного решения
    y_full = [y0] + list(y_unknown)
    return x, y_full

def exact_solution(x):
    return 2 + x + 2 * x * np.log(x)

# Решение для шага h=0.2
h1 = 0.2
x1, y_num1 = solve_bvp(h1)
y_exact1 = [exact_solution(xi) for xi in x1]
abs_error1 = [abs(num - ex) for num, ex in zip(y_num1, y_exact1)]
rel_error1 = [abs(err / ex) if ex != 0 else 0 for err, ex in zip(abs_error1, y_exact1)]



# Вывод результатов для h=0.2
print("="*80)
print(f"Решение с шагом h = {h1}")
print("-"*80)
print(f"{'x':<8}{'Численное':<15}{'Точное':<15}{'Абс. погр.':<15}{'Отн. погр. (%)':<15}")
print("-"*80)
for i in range(len(x1)):
    print(f"{x1[i]:<8.4f}{y_num1[i]:<15.6f}{y_exact1[i]:<15.6f}{abs_error1[i]:<15.6f}{rel_error1[i]*100:<15.6f}")


# Максимальные погрешности
max_abs_error1 = max(abs_error1)

print("\n" + "="*80)
print(f"Макс. абс. погр. (h={h1}): {max_abs_error1:.6f}")
