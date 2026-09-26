import numpy as np
import matplotlib.pyplot as plt

def divided_diff_table(x, y):
    n = len(x)
    table = [[0.0] * n for _ in range(n)]
    for i in range(n):
        table[i][0] = y[i]
    for j in range(1, n):
        for i in range(n - j):
            table[i][j] = (table[i + 1][j - 1] - table[i][j - 1]) / (x[i + j] - x[i])
    return table

def newton_poly(x_nodes, table, x_val):
    n = len(x_nodes)
    coefs = [table[0][j] for j in range(n)]
    result = coefs[0]
    p = 1.0
    for i in range(1, n):
        p *= (x_val - x_nodes[i - 1])
        result += coefs[i] * p
    return result

# Тестова функція типу Рунге, масштабована на робочий інтервал [a, b]
def runge_test_func(x_val, a, b):
    # Нормування x до діапазону [-1, 1] для відтворення класичного ефекту Рунге
    z = 2.0 * (x_val - a) / (b - a) - 1.0
    return 1.0 / (1.0 + 25.0 * (z ** 2))

# Вхідні межі згідно з Варіантом 1
x = [1000.0, 2000.0, 4000.0, 8000.0, 16000.0]

# =====================================================================
# 1. ВПЛИВ КРОКУ h НА ТОЧНІСТЬ (ФІКСОВАНИЙ ІНТЕРВАЛ)
# =====================================================================
a, b = x[0], x[-1]
node_counts = [5, 10, 20]

x_dense = np.linspace(a, b, num=400)
f_reference = np.array([runge_test_func(xv, a, b) for xv in x_dense])

plt.figure(figsize=(8, 5))
results_step = []

for n_nodes in node_counts:
    x_sub = list(np.linspace(a, b, n_nodes))
    y_sub = [runge_test_func(xv, a, b) for xv in x_sub]
    table_sub = divided_diff_table(x_sub, y_sub)
    
    y_interp = np.array([newton_poly(x_sub, table_sub, xv) for xv in x_dense])
    err = np.abs(y_interp - f_reference)
    max_err = err.max()
    results_step.append((n_nodes, max_err))
    
    print(f"n_nodes={n_nodes}: крок h={(b-a)/(n_nodes-1):.2f}, макс. похибка={max_err:.6e}")
    plt.plot(x_dense, err, label=f"{n_nodes} вузлів")

plt.xlabel('x')
plt.ylabel('|похибка|')
plt.yscale('log')
plt.title('Вплив кроку h на точність (фіксований інтервал)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# =====================================================================
# 2. ВПЛИВ РОЗШИРЕННЯ ІНТЕРВАЛУ НА ТОЧНІСТЬ (ФІКСОВАНИЙ КРОК)
# =====================================================================
h_fixed = (b - a) / 20.0  # базовий фіксований крок
print(f"\nФіксований крок h = {h_fixed:.3f}")

plt.figure(figsize=(8, 5))
results_interval = []

for n_nodes in node_counts:
    b_var = a + h_fixed * n_nodes
    b_var = min(b_var, x[-1])  # не виходимо за межі відомого інтервалу
    
    x_sub = list(np.linspace(a, b_var, n_nodes))
    y_sub = [runge_test_func(xv, a, b_var) for xv in x_sub]
    table_sub = divided_diff_table(x_sub, y_sub)
    
    x_dense_var = np.linspace(a, b_var, num=300)
    f_ref_var = np.array([runge_test_func(xv, a, b_var) for xv in x_dense_var])
    y_interp_var = np.array([newton_poly(x_sub, table_sub, xv) for xv in x_dense_var])
    
    err_var = np.abs(y_interp_var - f_ref_var)
    max_err = err_var.max()
    results_interval.append((n_nodes, b_var, max_err))
    
    print(f"n_nodes={n_nodes}: інтервал=[{a:.0f}, {b_var:.0f}], макс. похибка={max_err:.6e}")
    plt.plot(x_dense_var, err_var, label=f"{n_nodes} вузлів, b={b_var:.0f}")

plt.xlabel('x')
plt.ylabel('|похибка|')
plt.yscale('log')
plt.title('Вплив розширення інтервалу на точність (фіксований крок)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# =====================================================================
# 3. АНАЛІЗ ЕФЕКТУ РУНГЕ
# =====================================================================
print("\nАналіз ефекту Рунге:")
for n_nodes, max_err in results_step:
    print(f"  n={n_nodes:3d}  максимальна похибка = {max_err:.6e}")
