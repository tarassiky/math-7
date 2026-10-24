"""
Генерирует linear_algebra_lab.ipynb.
Запуск:
    python build_notebook.py
"""

import json


def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def code(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


cells = [
    md("# Лабораторная работа: линейная алгебра для Data Science\n\n"
       "Единая цепочка: вектор → матрица → линейное отображение → "
       "определитель и ранг → базис → системы → проекция → "
       "наименьшие квадраты → PCA → SVD."),

    md("## Общий набор данных\n\n"
       "| Студент | Часы | Упражнения | Балл |\n"
       "|---|---:|---:|---:|\n"
       "| Аня | 1 | 2 | 48 |\n"
       "| Борис | 2 | 2 | 52 |\n"
       "| Вера | 2 | 3 | 55 |\n"
       "| Глеб | 3 | 4 | 63 |\n"
       "| Дина | 4 | 5 | 70 |\n"
       "| Егор | 5 | 7 | 78 |\n"
       "| Жанна | 6 | 8 | 85 |\n"
       "| Илья | 7 | 10 | 94 |\n\n"
       "$$\\mathbf X\\in\\mathbb R^{8\\times2},\\quad \\mathbf y\\in\\mathbb R^8.$$"),

    code("import numpy as np\n"
         "import matplotlib.pyplot as plt\n\n"
         "np.set_printoptions(precision=4, suppress=True)\n\n"
         "STUDENT_NAMES = np.array([\"Аня\",\"Борис\",\"Вера\",\"Глеб\","
         "\"Дина\",\"Егор\",\"Жанна\",\"Илья\"])\n"
         "X = np.array([[1,2],[2,2],[2,3],[3,4],[4,5],[5,7],[6,8],[7,10]], "
         "dtype=float)\n"
         "y = np.array([48,52,55,63,70,78,85,94], dtype=float)\n"
         "print(\"X.shape =\", X.shape, \"; y.shape =\", y.shape)"),

    md("### Задание 1.1. Координаты и длина\n\n"
       "**[Бумага]** $\\overrightarrow{AB}=B-A=(2,-6,3)$, "
       "$|\\overrightarrow{AB}|=\\sqrt{4+36+9}=7$."),
    code("A = np.array([1,2,3], dtype=float)\n"
         "B = np.array([3,-4,6], dtype=float)\n"
         "AB = B - A\n"
         "length = np.linalg.norm(AB)\n"
         "assert np.allclose(AB, [2,-6,3]) and abs(length-7) < 1e-12\n"
         "print(\"AB =\", AB, \"; |AB| =\", length)"),

    md("### Задание 1.2. Угол между векторами\n\n"
       "$\\mathbf a=(-1,1,0)^{\\mathsf T}$, $\\mathbf b=(1,-2,2)^{\\mathsf T}$.\n\n"
       "**[Бумага]** $\\mathbf a^{\\mathsf T}\\mathbf b=-3$; "
       "$\\cos\\varphi=-1/\\sqrt2$; $\\varphi=135^\\circ$."),
    code("a = np.array([-1,1,0], dtype=float)\n"
         "b = np.array([1,-2,2], dtype=float)\n"
         "dot = a @ b\n"
         "cos_phi = dot / (np.linalg.norm(a)*np.linalg.norm(b))\n"
         "phi = np.degrees(np.arccos(np.clip(cos_phi, -1, 1)))\n"
         "assert np.isclose(dot,-3) and np.isclose(phi,135)\n"
         "print(f\"a·b={dot}, cos φ={cos_phi:.4f}, φ={phi:.2f}°\")"),

    md("### Задание 1.3. Ортогональность\n\n"
       "**[Бумага]** $3\\cdot2+2x-1\\cdot2=4+2x=0 \\Rightarrow x=-2$."),
    code("a = np.array([3,2,-1], dtype=float)\n"
         "x_value = -2.0\n"
         "b = np.array([2, x_value, 2], dtype=float)\n"
         "assert np.isclose(a @ b, 0)\n"
         "print(\"x =\", x_value, \"; a·b =\", a @ b)"),

    md("### Задание 1.4. Векторы объектов\n\n"
       "**[Бумага]** Илья − Аня $=(6,8)$, $\\|\\cdot\\|=10$."),
    code("delta = X[-1] - X[0]\n"
         "dist = np.linalg.norm(delta)\n"
         "assert np.allclose(delta, [6,8]) and np.isclose(dist, 10)\n"
         "print(\"Илья − Аня =\", delta, \"; норма =\", dist)"),

    md("## 2. Матрицы\n\n### Задание 2.1. Сложение матриц"),
    code("A = np.array([[3,5,7],[2,-1,0],[4,3,2]])\n"
         "B = np.array([[1,2,4],[2,3,-2],[-1,0,1]])\n"
         "S = A + B\n"
         "assert np.array_equal(S, np.array([[4,7,11],[4,2,-2],[3,3,3]]))\n"
         "print(\"A + B =\\n\", S)"),

    md("### Задание 2.2. Линейная комбинация"),
    code("A = np.array([[3,5],[4,1]])\n"
         "B = np.array([[2,3],[1,-2]])\n"
         "R = 2*A + 5*B\n"
         "assert np.array_equal(R, np.array([[16,25],[13,-8]]))\n"
         "print(\"2A + 5B =\\n\", R)"),

    md("### Задание 2.3. Произведение матриц\n\n"
       "$\\mathbf{AB}\\ne\\mathbf{BA}$ — умножение некоммутативно."),
    code("A = np.array([[1,3,1],[2,0,4],[1,2,3]])\n"
         "B = np.array([[2,1,0],[1,-1,2],[3,2,1]])\n"
         "AB = A @ B; BA = B @ A\n"
         "assert np.array_equal(AB, np.array([[8,0,7],[16,10,4],[13,5,7]]))\n"
         "assert np.array_equal(BA, np.array([[4,6,6],[1,7,3],[8,11,14]]))\n"
         "assert not np.array_equal(AB, BA)\n"
         "print(\"AB =\\n\", AB, \"\\nBA =\\n\", BA)"),

    md("### Задание 2.4. Размерность матрицы данных"),
    code("print(\"X.shape =\", X.shape, \"; y.shape =\", y.shape)\n"
         "print(\"Средние столбцов:\", X.mean(axis=0))\n"
         "print(\"X^T X =\\n\", X.T @ X)"),

    md("## 3. Линейные отображения\n\n### Задание 3.1. Матрица преобразования"),
    code("T = np.array([[0,1,-1],[1,0,0],[1,0,1]], dtype=float)\n"
         "for e in np.eye(3):\n"
         "    manual = np.array([e[1]-e[2], e[0], e[0]+e[2]], dtype=float)\n"
         "    assert np.allclose(T @ e, manual)\n"
         "print(\"T =\\n\", T)"),

    md("### Задание 3.2. Образ вектора"),
    code("A = np.array([[1,-1,0],[3,0,1],[1,1,-2]], dtype=float)\n"
         "a = np.array([1,-2,3], dtype=float)\n"
         "image = A @ a\n"
         "assert np.allclose(image, [3,6,-7])\n"
         "print(\"A a =\", image)"),

    md("## 4. Определитель и ранг\n\n### Задание 4.1. Определитель\n\n"
       "$\\det= -17\\ne0$ → обратима."),
    code("A = np.array([[5,2,3],[-4,3,5],[2,3,4]], dtype=float)\n"
         "det_A = np.linalg.det(A)\n"
         "assert np.isclose(det_A, -17, atol=1e-9)\n"
         "print(\"det A =\", det_A)"),

    md("### Задание 4.2. Ранг"),
    code("A1 = np.array([[1,2],[0,5],[-1,4],[1,1]], dtype=float)\n"
         "A2 = np.array([[3,-5,-1,-2],[8,-6,3,-7],[2,4,5,-3]], dtype=float)\n"
         "r1 = np.linalg.matrix_rank(A1)\n"
         "r2 = np.linalg.matrix_rank(A2)\n"
         "assert r1 == 2 and r2 == 2\n"
         "print(\"rank(A1) =\", r1, \"; rank(A2) =\", r2)"),

    md("## 5. Базис и координаты\n\n### Задание 5.1. Проверка базиса"),
    code("B = np.column_stack([np.array([0,1,2]), np.array([1,0,1]), "
         "np.array([-1,2,4])]).astype(float)\n"
         "rank = np.linalg.matrix_rank(B)\n"
         "det = np.linalg.det(B)\n"
         "assert rank == 2 and abs(det) < 1e-9\n"
         "print(\"rank =\", rank, \"; det =\", det)"),

    md("### Задание 5.2. Координаты в новом базисе"),
    code("p = np.array([2,1,-1], dtype=float)\n"
         "r = np.array([1,-1,1], dtype=float)\n"
         "q = np.array([0,3,2], dtype=float)\n"
         "x = np.array([1,-4,4], dtype=float)\n"
         "B = np.column_stack([p,r,q])\n"
         "coords = np.linalg.solve(B, x)\n"
         "assert np.allclose(coords, [-1,3,0])\n"
         "print(\"Координаты =\", coords)"),

    md("## 6. Системы\n\n### Задание 6.1. Квадратная система"),
    code("A = np.array([[2,-1,1],[3,2,2],[1,-2,1]], dtype=float)\n"
         "b = np.array([2,-2,1], dtype=float)\n"
         "x_sol = np.linalg.solve(A, b)\n"
         "residual = A @ x_sol - b\n"
         "assert np.allclose(x_sol, [2,-1,-3]) and np.allclose(residual, 0)\n"
         "print(\"x =\", x_sol, \"; невязка =\", residual)"),

    md("### Задание 6.2. Недоопределённая система\n\n"
       "$x_3=t$, $x_1=(13-46t)/7$, $x_2=50/7+t$."),
    code("A = np.array([[3,-4,6],[2,5,4]], dtype=float)\n"
         "b = np.array([-7,12], dtype=float)\n"
         "for t in (-2.0, 0.0, 3.0):\n"
         "    x_t = np.array([(13-46*t)/7, 50/7 + t, t], dtype=float)\n"
         "    assert np.allclose(A @ x_t - b, 0)\n"
         "    print(f\"t={t:+.1f}  x={x_t}  невязка={A @ x_t - b}\")"),

    md("## 7. Проекция\n\n### Задание 7.1. Проекция на прямую\n\n"
       "$\\widehat{\\mathbf y}=(2,2)^{\\mathsf T}$, "
       "$\\mathbf e=(1,-1)^{\\mathsf T}$, $\\mathbf a^{\\mathsf T}\\mathbf e=0$."),
    code("yv = np.array([3.0,1.0]); a = np.array([1.0,1.0])\n"
         "c = (a @ yv) / (a @ a)\n"
         "y_hat = c * a; e = yv - y_hat\n"
         "assert np.isclose(c, 2) and np.allclose(y_hat, [2,2]) "
         "and np.isclose(a @ e, 0)\n"
         "print(\"c =\", c, \"; y_hat =\", y_hat, \"; e =\", e)\n"
         "\n"
         "plt.figure(figsize=(5,5))\n"
         "o = np.zeros(2)\n"
         "plt.quiver(*o, *yv, angles='xy', scale_units='xy', scale=1, label='y')\n"
         "plt.quiver(*o, *y_hat, angles='xy', scale_units='xy', scale=1, label='ŷ')\n"
         "plt.quiver(*y_hat, *e, angles='xy', scale_units='xy', scale=1, label='e')\n"
         "plt.xlim(-0.5,3.5); plt.ylim(-0.5,3.5); plt.grid(); plt.legend()\n"
         "plt.gca().set_aspect('equal'); plt.show()"),

    md("## 8. Наименьшие квадраты\n\n### Задание 8.1. Регрессия"),
    code("D = np.column_stack([np.ones(len(X)), X])\n"
         "beta, *_ = np.linalg.lstsq(D, y, rcond=None)\n"
         "y_hat_reg = D @ beta\n"
         "print(\"beta =\", beta)\n"
         "print(\"y_hat =\", y_hat_reg)"),

    md("### Задание 8.2. Качество приближения"),
    code("residuals = y - y_hat_reg\n"
         "sse = float((residuals**2).sum())\n"
         "rmse = float(np.sqrt(sse / len(y)))\n"
         "r2 = 1 - sse / float(((y - y.mean())**2).sum())\n"
         "print(f\"SSE={sse:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}\")\n"
         "assert r2 > 0.99\n"
         "\n"
         "fig, axes = plt.subplots(1,2, figsize=(11,4))\n"
         "axes[0].scatter(y, y_hat_reg)\n"
         "lims = [min(y.min(), y_hat_reg.min())-2, max(y.max(), y_hat_reg.max())+2]\n"
         "axes[0].plot(lims, lims, 'r--')\n"
         "axes[0].set_xlabel('наблюдение'); axes[0].set_ylabel('предсказание')\n"
         "axes[1].bar(STUDENT_NAMES, residuals)\n"
         "axes[1].axhline(0, color='k', lw=0.5)\n"
         "axes[1].tick_params(axis='x', rotation=45)\n"
         "plt.tight_layout(); plt.show()"),

    md("### Задание 8.3. Регрессия как проекция"),
    code("orth = D.T @ (y - y_hat_reg)\n"
         "assert np.allclose(orth, 0, atol=1e-9)\n"
         "print(\"D^T (y - y_hat) =\", orth)"),

    md("## 9. PCA\n\n### Задание 9.1. Главные компоненты"),
    code("mean_X = X.mean(axis=0); scale_X = X.std(axis=0, ddof=0)\n"
         "Z = (X - mean_X) / scale_X\n"
         "C = Z.T @ Z / len(Z)\n"
         "eigvals, eigvecs = np.linalg.eigh(C)\n"
         "order = np.argsort(eigvals)[::-1]\n"
         "eigvals = eigvals[order]; eigvecs = eigvecs[:, order]\n"
         "ratio = eigvals / eigvals.sum()\n"
         "scores = Z @ eigvecs[:, :1]\n"
         "Z_rec = scores @ eigvecs[:, :1].T\n"
         "X_rec = Z_rec * scale_X + mean_X\n"
         "print(\"eigenvalues =\", eigvals)\n"
         "print(\"explained ratio =\", ratio)\n"
         "print(\"first component =\", eigvecs[:,0])\n"
         "assert ratio[0] > 0.99\n"
         "\n"
         "for i in np.argsort(scores[:,0]):\n"
         "    print(f\"  {STUDENT_NAMES[i]:8s}  score = {scores[i,0]:+.4f}\")\n"
         "\n"
         "fig, axes = plt.subplots(1,2, figsize=(11,4.5))\n"
         "axes[0].scatter(Z[:,0], Z[:,1])\n"
         "for name, pt in zip(STUDENT_NAMES, Z):\n"
         "    axes[0].annotate(name, pt, xytext=(4,4), "
         "textcoords='offset points', fontsize=8)\n"
         "line = np.linspace(-2.5,2.5,2)[:,None] * eigvecs[:,0]\n"
         "axes[0].plot(line[:,0], line[:,1], 'r-', label='PC1')\n"
         "axes[0].set_title('Стандартизованные данные')\n"
         "axes[0].axis('equal'); axes[0].legend(); axes[0].grid(alpha=0.3)\n"
         "axes[1].scatter(X[:,0], X[:,1], label='исходные')\n"
         "axes[1].scatter(X_rec[:,0], X_rec[:,1], label='восстановленные')\n"
         "for o_, r_ in zip(X, X_rec):\n"
         "    axes[1].plot([o_[0], r_[0]], [o_[1], r_[1]], color='0.7', lw=1)\n"
         "axes[1].set_title('Восстановление по одной компоненте')\n"
         "axes[1].legend(); axes[1].grid(alpha=0.3)\n"
         "plt.tight_layout(); plt.show()"),

    md("### Задание 9.2. Польза и ограничения"),
    code("corr = np.corrcoef(X.T)[0,1]\n"
         "rmse_rec = np.sqrt(np.mean((X - X_rec)**2))\n"
         "print(\"corr(X1,X2) =\", corr)\n"
         "print(\"RMSE восстановления =\", rmse_rec)"),

    md("## 10. SVD\n\n### Задание 10.1. Малоранговое приближение"),
    code("def make_demo_image(shape=(252,526), seed=0):\n"
         "    rng = np.random.default_rng(seed)\n"
         "    h,w = shape\n"
         "    yy,xx = np.mgrid[0:h,0:w]\n"
         "    base  = 0.5*np.exp(-((xx-w/3)**2+(yy-h/2)**2)/(2*80**2))\n"
         "    base += 0.3*np.exp(-((xx-2*w/3)**2+(yy-h/3)**2)/(2*50**2))\n"
         "    base += 0.15*np.sin(xx/20)*np.cos(yy/15)\n"
         "    base += 0.05*rng.standard_normal(shape)\n"
         "    return np.clip(base, 0, 1)\n"
         "\n"
         "A_img = make_demo_image()\n"
         "U, s, Vt = np.linalg.svd(A_img, full_matrices=False)\n"
         "m, n = A_img.shape\n"
         "errors, storages = [], []\n"
         "for k in [2, 5, 20, 50]:\n"
         "    Ak = (U[:, :k] * s[:k]) @ Vt[:k, :]\n"
         "    err = np.linalg.norm(A_img - Ak, 'fro') / np.linalg.norm(A_img, 'fro')\n"
         "    errors.append(err); storages.append(k*(m+n+1))\n"
         "    print(f\"k={k:2d}  ошибка={err:.5f}  "
         "хранить={k*(m+n+1):6d}  доля={k*(m+n+1)/(m*n):.2%}\")\n"
         "assert all(errors[i] >= errors[i+1]-1e-12 "
         "for i in range(len(errors)-1))"),

    md("### Задание 10.2. Связь PCA и SVD"),
    code("_, _, VtZ = np.linalg.svd(Z, full_matrices=False)\n"
         "align = abs(float(eigvecs[:,0] @ VtZ[0]))\n"
         "print(\"|cos|\", align)\n"
         "assert align > 0.999"),

    md("## 11. Итоговая интерпретация\n\n"
       "**Связный текст.** Вектор наблюдения задаёт один объект, "
       "матрица данных собирает весь набор строк. Линейное отображение "
       "переводит признаки в новое представление, а его матрица — "
       "векторы в образ. Ранг показывает число независимых направлений; "
       "det сообщает, не схлопнулось ли пространство. Система уравнений "
       "восстанавливает неизвестные параметры; если точного решения нет, "
       "проекция даёт ближайшую точку образа, а метод наименьших "
       "квадратов применяет эту проекцию к $\\mathbf y$ и находит "
       "коэффициенты. Главная компонента — направление максимальной "
       "дисперсии стандартизованных данных; сингулярное число измеряет "
       "масштаб направления, а малоранговое приближение оставляет "
       "несколько наибольших сингулярных чисел и уменьшает объём "
       "с малой ошибкой.\n\n"
       "**Регрессия:** вход — $\\mathbf D$ и $\\mathbf y$, выход — "
       "$\\beta$ и $\\hat{\\mathbf y}$; польза — линейное описание, "
       "риск — переобучение.\n\n"
       "**PCA:** вход — $\\mathbf X$, выход — направления и координаты; "
       "польза — сжатие и визуализация; риск — игнорируется $\\mathbf y$, "
       "линейность.\n\n"
       "**SVD:** вход — матрица $\\mathbf A$, выход — "
       "$\\mathbf U, \\mathbf S, \\mathbf V^{\\mathsf T}$ и приближение "
       "ранга $k$; польза — сжатие; риск — сильная усечка убирает детали."),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

with open("linear_algebra_lab.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)

print("Готово: linear_algebra_lab.ipynb")