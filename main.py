import csv
import math
import matplotlib.pyplot as plt

# =====================================================================
# 0. СТВОРЕННЯ ФАЙЛУ DATA.CSV (за Варіантом 1)
# =====================================================================
initial_data = [
    {"n": 1000, "t": 3.0},
    {"n": 2000, "t": 5.0},
    {"n": 4000, "t": 11.0},
    {"n": 8000, "t": 28.0},
    {"n": 16000, "t": 85.0}
]

with open("data.csv", mode="w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=["n", "t"])
    writer.writeheader()
    writer.writerows(initial_data)

# =====================================================================
# 1. ЗЧИТУВАННЯ ДАНИХ З CSV-ФАЙЛУ
# =====================================================================
def read_data(filename):
    x = []
    y = []
    with open(filename, "r", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            x.append(float(row["n"]))
            y.append(float(row["t"]))
    return x, y

x, y = read_data("data.csv")
print("1. Вхідні дані:")
print("   x:", x)
print("   y:", y)

# =====================================================================
# 2. ПОБУДОВА ТАБЛИЦІ РОЗДІЛЕНИХ РІЗНИЦЬ
# =====================================================================
def divided_diff_table(x, y):
    n = len(x)
    table = [[0.0] * n for _ in range(n)]
    for i in range(n):
        table[i][0] = y[i]
    for j in range(1, n):
        for i in range(n - j):
            table[i][j] = (table[i + 1][j - 1] - table[i][j - 1]) / (x[i + j] - x[i])
    return table

table_div = divided_diff_table(x, y)
print("\n2. Таблиця розділених різниць:")
for row in table_div:
    print(["%.6f" % v for v in row])

# =====================================================================
# 3. МЕТОДИ ІНТЕРПОЛЯЦІЇ: НЬЮТОНА ТА ФАКТОРІАЛЬНИЙ
# =====================================================================

# --- Метод 1: Многочлен Ньютона (для довільних вузлів) ---
coefs_newton = [table_div[0][j] for j in range(len(x))]

def newton_poly(x_val, x_nodes, coefs):
    n = len(coefs)
    result = coefs[0]
    p = 1.0
    for i in range(1, n):
        p *= (x_val - x_nodes[i - 1])
        result += coefs[i] * p
    return result

# --- Метод 2: Факторіальні многочлени (для рівновіддалених вузлів) ---
def finite_diff_table(y_vals):
    n = len(y_vals)
    diffs = [[0.0] * n for _ in range(n)]
    for i in range(n):
        diffs[i][0] = y_vals[i]
    for j in range(1, n):
        for i in range(n - j):
            diffs[i][j] = diffs[i + 1][j - 1] - diffs[i][j - 1]
    return diffs

def factorial_power(t_val, k):
    prod = 1.0
    for i in range(k):
        prod *= (t_val - i)
    return prod

def factorial_poly(x_val, x_nodes, y_nodes):
    h = x_nodes[1] - x_nodes[0]
    t = (x_val - x_nodes[0]) / h
    diffs = finite_diff_table(y_nodes)
    
    result = diffs[0][0]
    for k in range(1, len(y_nodes)):
        delta_k = diffs[0][k]
        result += (delta_k / math.factorial(k)) * factorial_power(t, k)
    return result

# 1. Прогноз методом Ньютона для точки n = 6000
target_n = 6000
pred_newton = newton_poly(target_n, x, coefs_newton)

# 2. Побудова допоміжної рівномірної сітки для факторіального методу
num_uniform = 5
h_uniform = (x[-1] - x[0]) / (num_uniform - 1)
x_uniform = [x[0] + i * h_uniform for i in range(num_uniform)]
y_uniform = [newton_poly(xi, x, coefs_newton) for xi in x_uniform]

# 3. Прогноз факторіальним методом на рівномірній сітці
pred_factorial = factorial_poly(target_n, x_uniform, y_uniform)

print(f"\n3. Прогноз часу виконання при n = {target_n}:")
print(f"   - Метод Ньютона (розділені різниці):  {pred_newton:.4f} мс")
print(f"   - Метод факторіальних многочленів:    {pred_factorial:.4f} мс")

# =====================================================================
# 4. ПОБУДОВА ГРАФІКІВ
# =====================================================================
num_pts = 300
x_min, x_max = min(x), max(x)
step = (x_max - x_min) / (num_pts - 1)
x_curve = [x_min + i * step for i in range(num_pts)]

y_newton_curve = [newton_poly(xi, x, coefs_newton) for xi in x_curve]
y_factorial_curve = [factorial_poly(xi, x_uniform, y_uniform) for xi in x_curve]

plt.figure(figsize=(10, 6))
plt.plot(x_curve, y_newton_curve, label="Многочлен Ньютона", color="blue", linewidth=2)
plt.plot(x_curve, y_factorial_curve, label="Факторіальний многочлен", color="orange", linestyle="--")
plt.scatter(x, y, color="red", zorder=5, label="Експериментальні точки")
plt.scatter([target_n], [pred_newton], color="green", marker="^", s=110, zorder=6,
            label=f"Прогноз n=6000 ({pred_newton:.2f} мс)")

plt.title("Інтерполяція продуктивності алгоритму (Варіант 1)")
plt.xlabel("Розмір вхідних даних n")
plt.ylabel("Час виконання t (мс)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()

# =====================================================================
# 5. ДОСЛІДЖЕННЯ ПОХИБОК (5, 10, 20 ВУЗЛІВ) ТА ЕФЕКТ РУНГЕ
# =====================================================================
def f_model(val):
    return 0.000363 * val * math.log2(val) - 0.618

node_counts = [5, 10, 20]
plt.figure(figsize=(11, 6))

for k in node_counts:
    h_k = (16000 - 1000) / (k - 1)
    x_k = [1000 + i * h_k for i in range(k)]
    y_k = [f_model(xi) for xi in x_k]
    
    tbl_k = divided_diff_table(x_k, y_k)
    c_k = [tbl_k[0][j] for j in range(k)]
    
    errors = [abs(f_model(xi) - newton_poly(xi, x_k, c_k)) for xi in x_curve]
    plt.plot(x_curve, errors, label=f"Похибка при n = {k} вузлах")

plt.yscale("log")
plt.title("Дослідження абсолютної похибки при різній кількості вузлів")
plt.xlabel("Розмір вхідних даних n")
plt.ylabel("Похибка |f(x) - N(x)| (мс)")
plt.legend()
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.show()
