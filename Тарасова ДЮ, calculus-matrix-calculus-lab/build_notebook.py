"""
Генерирует calculus_matrix_calculus_lab.ipynb.
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
    md("# Лабораторная работа: математический анализ и матричное исчисление\n\n"
       "Единая цепочка: суммы и индексы → производная одной переменной → "
       "градиент и дифференциал → производная по направлению → якобиан, "
       "JVP, VJP → гессиан → матричное дифференцирование → композиция "
       "с МНК и сигмоидой → численная оптимизация."),

    md("## Общий пакет данных\n\n"
       "$N=4$, $D=3$, $K=2$, $\\lambda=0.05$."),
    code("import numpy as np\n"
         "import matplotlib.pyplot as plt\n\n"
         "np.set_printoptions(precision=6, suppress=True)\n\n"
         "X_data = np.array([\n"
         "    [1, 0, -1],\n"
         "    [0, 1, 1],\n"
         "    [1, 1, 0],\n"
         "    [-1, 2, 1],\n"
         "], dtype=float)\n"
         "Y_data = np.array([\n"
         "    [0.2, 0.8],\n"
         "    [0.7, 0.1],\n"
         "    [0.6, 0.4],\n"
         "    [0.1, 0.9],\n"
         "], dtype=float)\n"
         "W_initial = np.array([\n"
         "    [0.2, -0.3],\n"
         "    [0.4, 0.1],\n"
         "    [-0.5, 0.2],\n"
         "], dtype=float)\n"
         "b_initial = np.array([0.1, -0.2], dtype=float)\n"
         "lam = 0.05\n\n"
         "def sigmoid(z):\n"
         "    return 1.0 / (1.0 + np.exp(-np.clip(z, -50.0, 50.0)))\n\n"
         "print(\"X:\", X_data.shape, \"Y:\", Y_data.shape,\n"
         "      \"W:\", W_initial.shape, \"b:\", b_initial.shape, \"lam:\", lam)"),

    # ---------- 1.1 ----------
    md("## 1. Суммы и индексы\n\n### Задание 1.1. От суммы к матрице\n\n"
       "$\\mathbf A\\in\\mathbb R^{2\\times3}$, $\\mathbf B\\in\\mathbb R^{3\\times2}$, "
       "$\\mathbf C=\\mathbf A\\mathbf B\\in\\mathbb R^{2\\times2}$, "
       "$C_{ij}=\\sum_k A_{ik}B_{kj}$."),
    code("A = np.array([[1,2,-1],[0,3,1]], dtype=float)\n"
         "B = np.array([[2,1],[1,-1],[0,2]], dtype=float)\n\n"
         "C_ref = np.array([[4,-3],[3,-1]], dtype=float)\n"
         "assert np.allclose(A @ B, C_ref)\n\n"
         "C_loops = np.zeros((2, 2))\n"
         "for i in range(2):\n"
         "    for j in range(2):\n"
         "        s = 0.0\n"
         "        for k in range(3):\n"
         "            s += A[i, k] * B[k, j]\n"
         "        C_loops[i, j] = s\n"
         "assert np.allclose(C_loops, C_ref)\n"
         "print(\"A @ B =\\n\", A @ B, \"\\nC_loops =\\n\", C_loops)"),

    # ---------- 1.2 ----------
    md("### Задание 1.2. Три индекса и две свёртки\n\n"
       "$L_{nc}=\\sum_{t}\\sum_{k}\\sum_{d}S_{ntk}W_{kd}V_{dc}$. "
       "Свободные: $n,c$; суммируемые: $t,k,d$."),
    code("S_tensor = np.array([\n"
         "    [[1, 0, 0], [0, 1, 0]],\n"
         "    [[0, 1, 0], [0, 0, 1]],\n"
         "], dtype=float)\n"
         "W_tensor = np.array([[1, 0], [0, 2], [1, -1]], dtype=float)\n"
         "V_tensor = np.array([[1, -1], [2, 1]], dtype=float)\n"
         "print(\"S:\", S_tensor.shape, \"W:\", W_tensor.shape, \"V:\", V_tensor.shape)"),

    # ---------- 1.3 ----------
    md("### Задание 1.3. Циклы, матрицы и `einsum`"),
    code("L_loops = np.zeros((2, 2))\n"
         "for n_ in range(2):\n"
         "    for c_ in range(2):\n"
         "        s = 0.0\n"
         "        for t_ in range(2):\n"
         "            for k_ in range(3):\n"
         "                for d_ in range(2):\n"
         "                    s += S_tensor[n_, t_, k_] * W_tensor[k_, d_] * V_tensor[d_, c_]\n"
         "        L_loops[n_, c_] = s\n\n"
         "E_matrix = np.einsum('ntk,kd->nd', S_tensor, W_tensor)\n"
         "L_matrix = E_matrix @ V_tensor\n\n"
         "L_einsum_once = np.einsum('ntk,kd,dc->nc', S_tensor, W_tensor, V_tensor)\n"
         "L_einsum_twice = np.einsum('nc->nc', np.einsum('ntk,kd->nd', S_tensor, W_tensor) @ V_tensor)\n\n"
         "assert np.allclose(L_loops, L_matrix)\n"
         "assert np.allclose(L_matrix, L_einsum_once)\n"
         "assert np.allclose(L_einsum_once, L_einsum_twice)\n"
         "print(\"L_nc =\\n\", L_matrix)"),

    # ---------- 2.1 ----------
    md("## 2. Функции одной переменной\n\n"
       "### Задание 2.1. Производная как предел и как численный оператор\n\n"
       "Для $f(x)=e^x$, $x_0=1$: $f'(x_0)=e$."),
    code("x0 = 1.0\n"
         "h_values = np.logspace(-1, -16, 16)\n"
         "exact = np.exp(x0)\n\n"
         "D_plus  = (np.exp(x0 + h_values) - np.exp(x0)) / h_values\n"
         "D_centr = (np.exp(x0 + h_values/2) - np.exp(x0 - h_values/2)) / h_values\n"
         "D_minus = (np.exp(x0) - np.exp(x0 - h_values)) / h_values\n\n"
         "err_plus  = np.abs(D_plus  - exact)\n"
         "err_centr = np.abs(D_centr - exact)\n"
         "err_minus = np.abs(D_minus - exact)\n\n"
         "assert err_centr.min() < err_plus.min()\n"
         "print(f\"exact={exact:.6f}, лучший h для центральной: \"\n"
         "      f\"{h_values[np.argmin(err_centr)]:.1e}\")\n\n"
         "plt.figure(figsize=(8,4))\n"
         "plt.loglog(h_values, err_plus,  'o-', label='D_+ (прямая)')\n"
         "plt.loglog(h_values, err_centr, 's-', label='D_0 (центральная)')\n"
         "plt.loglog(h_values, err_minus, '^-', label='D_- (обратная)')\n"
         "plt.gca().invert_xaxis()\n"
         "plt.xlabel('h'); plt.ylabel('|ошибка|'); plt.legend(); plt.grid(True, which='both')\n"
         "plt.show()"),

    # ---------- 2.2 ----------
    md("### Задание 2.2. Правила дифференцирования\n\n"
       "1) $y=2x^3-5x^2+7x+4$, $y'=6x^2-10x+7$; в $x=1$: 3.\n\n"
       "2) $y=x^2e^x$, $y'=(2x+x^2)e^x$; в $x=1$: $3e$.\n\n"
       "3) $y=(2x^2+5)^2$, $y'=8x(2x^2+5)$; в $x=1$: 56.\n\n"
       "4) $y=\\ln(2x^3+3x^2)$, $y'=\\dfrac{6x^2+6x}{2x^3+3x^2}$; в $x=1$: 12/5=2.4."),
    code("x0 = 1.0\n"
         "h = 1e-5\n\n"
         "funcs = [\n"
         "    (lambda x: 2*x**3 - 5*x**2 + 7*x + 4,\n"
         "     lambda x: 6*x**2 - 10*x + 7),\n"
         "    (lambda x: x**2 * np.exp(x),\n"
         "     lambda x: (2*x + x**2) * np.exp(x)),\n"
         "    (lambda x: (2*x**2 + 5)**2,\n"
         "     lambda x: 8*x*(2*x**2 + 5)),\n"
         "    (lambda x: np.log(2*x**3 + 3*x**2),\n"
         "     lambda x: (6*x**2 + 6*x) / (2*x**3 + 3*x**2)),\n"
         "]\n\n"
         "for k, (f, df) in enumerate(funcs, 1):\n"
         "    central = (f(x0+h) - f(x0-h)) / (2*h)\n"
         "    exact = df(x0)\n"
         "    assert abs(central - exact) < 1e-4\n"
         "    print(f\"{k}: f'={exact:.6f}, центральная={central:.6f}\")"),

    # ---------- 2.3 ----------
    md("### Задание 2.3. Дифференциал и `max(0,x)`\n\n"
       "Для $y=\\ln(1+x^2)$: $y'=2x/(1+x^2)$, $dy=\\frac{2x}{1+x^2}dx$.\n\n"
       "В $x=1$, $dx=0.02$: $dy = 0.02$. Точное $\\Delta y = \\ln(1+1.02^2)-\\ln 2$.\n\n"
       "$r(x)=\\max(0,x)$: $r'_-(0)=0$, $r'_+(0)=1$, производной нет."),
    code("x0, dx = 1.0, 0.02\n"
         "dy_lin = (2*x0 / (1 + x0**2)) * dx\n"
         "dy_exact = np.log(1 + (x0+dx)**2) - np.log(1 + x0**2)\n"
         "print(f\"dy (линейно) = {dy_lin:.6f}, точное Δy = {dy_exact:.6f}\")\n\n"
         "relu = lambda x: np.maximum(0.0, x)\n"
         "h = 1e-5\n"
         "left  = (relu(0) - relu(-h)) / h\n"
         "right = (relu(h) - relu(0)) / h\n"
         "central = (relu(h) - relu(-h)) / (2*h)\n"
         "print(f\"r'_-(0)={left}, r'_+(0)={right}, центральная={central}\")"),

    # ---------- 3.1 ----------
    md("## 3. Частные производные и градиент\n\n"
       "### Задание 3.1. Полный дифференциал многочлена\n\n"
       "$z=5x^3+2x^2y^4-6x+9y$.\n\n"
       "$\\partial z/\\partial x = 15x^2+4xy^4-6$; "
       "$\\partial z/\\partial y = 8x^2y^3+9$.\n\n"
       "В $(1,-1)$: $\\nabla z=(13,-13)^\\top$."),
    code("def z_poly(p):\n"
         "    x, y = p\n"
         "    return 5*x**3 + 2*x**2*y**4 - 6*x + 9*y\n\n"
         "def grad_z(p):\n"
         "    x, y = p\n"
         "    return np.array([15*x**2 + 4*x*y**4 - 6,\n"
         "                     8*x**2*y**3 + 9])\n\n"
         "def central_gradient(f, p, h=1e-6):\n"
         "    g = np.zeros_like(p, dtype=float)\n"
         "    for i in range(p.size):\n"
         "        ep = np.zeros_like(p); ep[i] = h\n"
         "        g[i] = (f(p+ep) - f(p-ep)) / (2*h)\n"
         "    return g\n\n"
         "p0 = np.array([1.0, -1.0])\n"
         "g_exact = grad_z(p0)\n"
         "g_num = central_gradient(z_poly, p0)\n"
         "assert np.allclose(g_exact, [13, -13])\n"
         "assert np.allclose(g_exact, g_num, atol=1e-5)\n"
         "print(\"∇z(1,-1) =\", g_exact)"),

    # ---------- 3.2 ----------
    md("### Задание 3.2. Локальная плоскость\n\n"
       "$f(x,y)=x^2+xy+2y^2-3x+1$, $x_0=(1,-1)^\\top$.\n\n"
       "$\\nabla f=(2x+y-3,\\, x+4y)^\\top$; $\\nabla f(x_0)=(0,-3)^\\top$."),
    code("def f_quad(p):\n"
         "    x, y = p\n"
         "    return x**2 + x*y + 2*y**2 - 3*x + 1\n\n"
         "def grad_quad(p):\n"
         "    x, y = p\n"
         "    return np.array([2*x + y - 3, x + 4*y])\n\n"
         "p0 = np.array([1.0, -1.0])\n"
         "delta = np.array([0.01, -0.02])\n"
         "g0 = grad_quad(p0)\n"
         "lin = float(g0 @ delta)\n"
         "exact = f_quad(p0 + delta) - f_quad(p0)\n"
         "assert np.allclose(g0, [0, -3])\n"
         "print(f\"линейная поправка={lin:.6f}, точное Δf={exact:.6f}\")\n\n"
         "xx, yy = np.meshgrid(np.linspace(-1, 3, 60), np.linspace(-3, 1, 60))\n"
         "zz = xx**2 + xx*yy + 2*yy**2 - 3*xx + 1\n"
         "plt.figure(figsize=(5,5))\n"
         "plt.contour(xx, yy, zz, 20, cmap='viridis')\n"
         "plt.quiver(*p0, *g0, color='red', angles='xy', scale_units='xy', scale=1, label='∇f')\n"
         "plt.plot(*p0, 'ko'); plt.legend(); plt.grid(alpha=0.3); plt.axis('equal')\n"
         "plt.show()"),

    # ---------- 3.3 ----------
    md("### Задание 3.3. Численный градиент\n\n"
       "Число обращений к $f$: $2n$."),
    code("for h in [1e-3, 1e-5, 1e-7]:\n"
         "    gn = central_gradient(f_quad, p0, h)\n"
         "    print(f\"h={h:.0e}: |g_exact - g_num| = {np.linalg.norm(gn - grad_quad(p0)):.3e}\")"),

    # ---------- 4.1 ----------
    md("## 4. Направление и полная производная\n\n"
       "### Задание 4.1. Производная по направлению\n\n"
       "$v=(3/5,4/5)^\\top$, $\\nabla f(x_0)=(0,-3)^\\top$, "
       "$D_v f= -12/5 = -2.4$ < 0 — убывает."),
    code("v = np.array([3.0, 4.0]) / 5.0\n"
         "g0 = grad_quad(p0)\n"
         "Dv = float(g0 @ v)\n"
         "h = 1e-6\n"
         "Dv_num = (f_quad(p0 + h*v) - f_quad(p0 - h*v)) / (2*h)\n"
         "assert abs(Dv - Dv_num) < 1e-5\n"
         "gmax = g0 / np.linalg.norm(g0)\n"
         "print(f\"D_v f = {Dv:.6f}, направление max роста = {gmax}, |∇f|={np.linalg.norm(g0):.4f}\")"),

    # ---------- 4.2 ----------
    md("### Задание 4.2. Полная производная\n\n"
       "$F(t,x,y)=tx+x^2+xy+2y^2$, $x(t)=t$, $y(t)=t^2$.\n\n"
       "$\\frac{dF}{dt}=x + t\\frac{dx}{dt} + (2x+y)\\frac{dx}{dt} + (x+4y)\\frac{dy}{dt}$.\n\n"
       "Прямая подстановка: $F(t)=t^2+t^2+t^3+2t^4=2t^2+t^3+2t^4$, "
       "$F'(t)=4t+3t^2+8t^3$."),
    code("def F_total(t, x, y):\n"
         "    return t*x + x**2 + x*y + 2*y**2\n\n"
         "def dF_dt_full(t):\n"
         "    x, y = t, t**2\n"
         "    dx, dy = 1.0, 2*t\n"
         "    return x + t*dx + (2*x + y)*dx + (x + 4*y)*dy\n\n"
         "def dF_dt_subst(t):\n"
         "    return 4*t + 3*t**2 + 8*t**3\n\n"
         "h = 1e-6\n"
         "t0 = 1.0\n"
         "num = (F_total(t0+h, t0+h, (t0+h)**2) - F_total(t0-h, t0-h, (t0-h)**2)) / (2*h)\n"
         "assert abs(dF_dt_full(t0) - num) < 1e-5\n"
         "assert abs(dF_dt_full(t0) - dF_dt_subst(t0)) < 1e-12\n"
         "print(f\"dF/dt (полная) = {dF_dt_full(t0):.6f}, численно = {num:.6f}\")"),

    # ---------- 5.1 ----------
    md("## 5. Якобиан, JVP, VJP\n\n"
       "### Задание 5.1. Якобиан векторной функции\n\n"
       "$\\mathbf f(x_1,x_2)=(x_1^2+x_2,\\, e^{x_1-x_2},\\, x_1x_2)^\\top$, "
       "якобиан $3\\times2$."),
    code("def vec_f(p):\n"
         "    x1, x2 = p\n"
         "    return np.array([x1**2 + x2, np.exp(x1 - x2), x1*x2])\n\n"
         "def jac_f(p):\n"
         "    x1, x2 = p\n"
         "    return np.array([\n"
         "        [2*x1, 1.0],\n"
         "        [np.exp(x1 - x2), -np.exp(x1 - x2)],\n"
         "        [x2, x1],\n"
         "    ])\n\n"
         "xs = np.array([1.0, 0.0])\n"
         "J = jac_f(xs)\n"
         "print(\"f(x*) =\", vec_f(xs))\n"
         "print(\"J_f(x*) =\\n\", J)\n\n"
         "# Проверка каждой строки центральными разностями.\n"
         "h = 1e-6\n"
         "for i in range(3):\n"
         "    gi = np.zeros(2)\n"
         "    for j in range(2):\n"
         "        e = np.zeros(2); e[j] = h\n"
         "        gi[j] = (vec_f(xs+e)[i] - vec_f(xs-e)[i]) / (2*h)\n"
         "    assert np.allclose(gi, J[i], atol=1e-5)"),

    # ---------- 5.2 ----------
    md("### Задание 5.2. JVP\n\n"
       "$v=(1,-2)^\\top$."),
    code("v = np.array([1.0, -2.0])\n"
         "J = jac_f(xs)\n"
         "jvp_exact = J @ v\n"
         "jvp_num_list = []\n"
         "hs = np.logspace(-2, -8, 7)\n"
         "for h in hs:\n"
         "    jvp_num = (vec_f(xs + h*v/2) - vec_f(xs - h*v/2)) / h\n"
         "    jvp_num_list.append(np.linalg.norm(jvp_num - jvp_exact))\n"
         "print(\"JVP =\", jvp_exact)\n"
         "assert min(jvp_num_list) < 1e-4"),

    # ---------- 5.3 ----------
    md("### Задание 5.3. VJP и производная композиции\n\n"
       "$u=(1,2,-1)^\\top$, $c=(0,1,1)^\\top$, "
       "$h(x)=g(f(x))$."),
    code("u = np.array([1.0, 2.0, -1.0])\n"
         "c = np.array([0.0, 1.0, 1.0])\n"
         "vjp = u @ jac_f(xs)\n"
         "print(\"u^T J =\", vjp)\n\n"
         "def h_composite(p):\n"
         "    r = vec_f(p) - c\n"
         "    return 0.5 * r @ r\n\n"
         "def grad_h(p):\n"
         "    r = vec_f(p) - c\n"
         "    return jac_f(p).T @ r\n\n"
         "g_ana = grad_h(xs)\n"
         "g_num = central_gradient(h_composite, xs, 1e-6)\n"
         "assert np.allclose(g_ana, g_num, atol=1e-5)\n"
         "print(\"∇h(x*) аналитически =\", g_ana)\n"
         "print(\"∇h(x*) численно      =\", g_num)"),

    # ---------- 6.1 ----------
    md("## 6. Гессиан и кривизна\n\n"
       "### Задание 6.1. Минимум\n\n"
       "$z=2x^2+y^2+x-3y+2xy$.\n\n"
       "$\\nabla z=(4x+2y+1,\\, 2x+2y-3)^\\top=0 \\Rightarrow x=-2,y=3.5$."),
    code("def z_min(p):\n"
         "    x, y = p\n"
         "    return 2*x**2 + y**2 + x - 3*y + 2*x*y\n\n"
         "def grad_min(p):\n"
         "    x, y = p\n"
         "    return np.array([4*x + 2*y + 1, 2*x + 2*y - 3])\n\n"
         "H_min = np.array([[4.0, 2.0], [2.0, 2.0]])\n"
         "x_star = np.linalg.solve(H_min, -np.array([1.0, -3.0]))\n"
         "eigvals = np.linalg.eigvalsh(H_min)\n"
         "assert np.allclose(grad_min(x_star), 0)\n"
         "assert np.all(eigvals > 0)\n"
         "print(f\"стационарная точка x*={x_star}, собственные значения={eigvals}\")"),

    # ---------- 6.2 ----------
    md("### Задание 6.2. Седловая точка\n\n"
       "$z=x^2-2y^2-4xy+3x$, гессиан неопределённый."),
    code("def z_saddle(p):\n"
         "    x, y = p\n"
         "    return x**2 - 2*y**2 - 4*x*y + 3*x\n\n"
         "H_saddle = np.array([[2.0, -4.0], [-4.0, -4.0]])\n"
         "eigvals_s = np.linalg.eigvalsh(H_saddle)\n"
         "assert eigvals_s.min() < 0 < eigvals_s.max()\n"
         "print(\"собственные значения гессиана:\", eigvals_s)"),

    # ---------- 6.3 ----------
    md("### Задание 6.3. Квадратичная модель и HVP\n\n"
       "$\\varphi(x,y)=e^{x+2y}+x^2$, $x_0=(0,0)^\\top$."),
    code("def phi(p):\n"
         "    x, y = p\n"
         "    return np.exp(x + 2*y) + x**2\n\n"
         "def phi_grad(p):\n"
         "    x, y = p\n"
         "    e = np.exp(x + 2*y)\n"
         "    return np.array([e + 2*x, 2*e])\n\n"
         "def phi_hess(p):\n"
         "    x, y = p\n"
         "    e = np.exp(x + 2*y)\n"
         "    return np.array([[e + 2, 2*e], [2*e, 4*e]])\n\n"
         "x0 = np.zeros(2)\n"
         "g0 = phi_grad(x0)\n"
         "H0 = phi_hess(x0)\n"
         "assert np.allclose(g0, [1, 2])\n"
         "assert np.allclose(H0, [[3, 2], [2, 4]])\n\n"
         "for scale in [0.1, 0.05, 0.01]:\n"
         "    delta = scale*np.array([1.0, 1.0])\n"
         "    exact = phi(x0 + delta) - phi(x0)\n"
         "    lin = g0 @ delta\n"
         "    quad = lin + 0.5 * delta @ H0 @ delta\n"
         "    print(f\"scale={scale}: exact={exact:.6f}, lin={lin:.6f}, quad={quad:.6f}\")"),

    # ---------- 7.1 ----------
    md("## 7. Матричное дифференцирование\n\n"
       "### Задание 7.1. Линейно-квадратичная форма\n\n"
       "$f(x)=a^\\top x + x^\\top A x$, $\\nabla f = a + (A+A^\\top)x$, "
       "$H = A+A^\\top$."),
    code("A_q = np.array([[2.0, -1.0], [3.0, 1.0]])\n"
         "a_q = np.array([1.0, -2.0])\n"
         "x_q = np.array([2.0, -1.0])\n\n"
         "val = float(a_q @ x_q + x_q @ A_q @ x_q)\n"
         "grad = a_q + (A_q + A_q.T) @ x_q\n"
         "H = A_q + A_q.T\n"
         "assert np.allclose(grad, central_gradient(\n"
         "    lambda p: float(a_q @ p + p @ A_q @ p), x_q, 1e-6), atol=1e-5)\n"
         "print(\"f(x) =\", val, \"\\n∇f =\", grad, \"\\nH =\\n\", H)"),

    # ---------- 7.2 ----------
    md("### Задание 7.2. Производная по матрице\n\n"
       "$s(M)=a^\\top M b$, $\\partial s/\\partial M = a b^\\top$."),
    code("a_m = np.array([1.0, -2.0])\n"
         "b_m = np.array([3.0, 1.0, -1.0])\n"
         "M = np.array([[1.0, 2.0, 0.0], [-1.0, 3.0, 4.0]])\n\n"
         "s_val = float(a_m @ M @ b_m)\n"
         "grad_M = np.outer(a_m, b_m)\n"
         "h = 1e-6\n"
         "num_grad_M = np.zeros_like(M)\n"
         "for i in range(M.shape[0]):\n"
         "    for j in range(M.shape[1]):\n"
         "        Mp = M.copy(); Mp[i,j] += h\n"
         "        Mm = M.copy(); Mm[i,j] -= h\n"
         "        num_grad_M[i,j] = (a_m @ Mp @ b_m - a_m @ Mm @ b_m) / (2*h)\n"
         "assert np.allclose(grad_M, num_grad_M, atol=1e-5)\n"
         "print(\"s =\", s_val, \"\\n∂s/∂M =\\n\", grad_M)"),

    # ---------- 7.3 ----------
    md("### Задание 7.3. Матричный МНК-градиент\n\n"
       "$Q(W)=\\frac{1}{2N}\\|XW-Y\\|_F^2$, "
       "$\\nabla_W Q = \\frac 1N X^\\top(XW-Y)$, "
       "HVP: $H[\\Delta W]=\\frac 1N X^\\top X \\Delta W$."),
    code("def Q(W):\n"
         "    R = X_data @ W - Y_data\n"
         "    return float((R*R).sum()) / (2*len(X_data))\n\n"
         "def dQ(W):\n"
         "    return X_data.T @ (X_data @ W - Y_data) / len(X_data)\n\n"
         "g_W = dQ(W_initial)\n"
         "h = 1e-6\n"
         "num_g = np.zeros_like(W_initial)\n"
         "for i in range(W_initial.shape[0]):\n"
         "    for j in range(W_initial.shape[1]):\n"
         "        Wp = W_initial.copy(); Wp[i,j] += h\n"
         "        Wm = W_initial.copy(); Wm[i,j] -= h\n"
         "        num_g[i,j] = (Q(Wp) - Q(Wm)) / (2*h)\n"
         "assert np.allclose(g_W, num_g, atol=1e-6)\n"
         "print(\"Q(W0) =\", Q(W_initial), \"\\n∇_W Q =\\n\", g_W)"),

    # ---------- 8.1 ----------
    md("## 8. Сквозная композиция\n\n"
       "### Задание 8.1. Прямое вычисление и размерности\n\n"
       "$Z=XW+\\mathbf 1_N b^\\top$, $H=\\sigma(Z)$, "
       "$\\mathcal L=\\frac{1}{2N}\\|H-Y\\|_F^2+\\frac\\lambda2\\|W\\|_F^2$."),
    code("Z = X_data @ W_initial + np.ones((len(X_data), 1)) @ b_initial.reshape(1, -1)\n"
         "H = sigmoid(Z)\n"
         "L_main = np.sum((H - Y_data)**2) / (2*len(X_data))\n"
         "L_reg  = lam / 2 * np.sum(W_initial**2)\n"
         "print(\"L_data =\", L_main, \"; L_reg =\", L_reg, \"; L_total =\", L_main + L_reg)"),

    # ---------- 8.2 ----------
    md("### Задание 8.2. Обратный проход\n\n"
       "$G_H = \\frac 1N R$, "
       "$G_Z = G_H\\odot H\\odot(1-H)$, "
       "$\\nabla_W\\mathcal L = X^\\top G_Z+\\lambda W$, "
       "$\\nabla_b\\mathcal L = G_Z^\\top\\mathbf 1_N$."),
    code("def loss_and_grads(W, b):\n"
         "    Z = X_data @ W + np.outer(np.ones(len(X_data)), b)\n"
         "    H = sigmoid(Z)\n"
         "    R = H - Y_data\n"
         "    L = float((R*R).sum()) / (2*len(X_data)) + lam/2 * (W*W).sum()\n"
         "    G_H = R / len(X_data)\n"
         "    G_Z = G_H * H * (1 - H)\n"
         "    gW = X_data.T @ G_Z + lam * W\n"
         "    gb = G_Z.sum(axis=0)\n"
         "    return L, gW, gb\n\n"
         "L_val, gW_val, gb_val = loss_and_grads(W_initial, b_initial)\n"
         "print(\"L =\", L_val)\n"
         "print(\"∇W =\\n\", gW_val)\n"
         "print(\"∇b =\", gb_val)"),

    # ---------- 8.3 ----------
    md("### Задание 8.3. Проверка конечными разностями"),
    code("h = 1e-6\n"
         "num_W = np.zeros_like(W_initial)\n"
         "for i in range(W_initial.shape[0]):\n"
         "    for j in range(W_initial.shape[1]):\n"
         "        Wp = W_initial.copy(); Wp[i,j] += h\n"
         "        Wm = W_initial.copy(); Wm[i,j] -= h\n"
         "        num_W[i,j] = (loss_and_grads(Wp, b_initial)[0]\n"
         "                      - loss_and_grads(Wm, b_initial)[0]) / (2*h)\n"
         "num_b = np.zeros_like(b_initial)\n"
         "for k in range(b_initial.size):\n"
         "    bp = b_initial.copy(); bp[k] += h\n"
         "    bm = b_initial.copy(); bm[k] -= h\n"
         "    num_b[k] = (loss_and_grads(W_initial, bp)[0]\n"
         "                - loss_and_grads(W_initial, bm)[0]) / (2*h)\n\n"
         "err_W = np.max(np.abs(num_W - gW_val))\n"
         "err_b = np.max(np.abs(num_b - gb_val))\n"
         "assert err_W < 1e-6 and err_b < 1e-6\n"
         "print(f\"max|ΔW| = {err_W:.2e}, max|Δb| = {err_b:.2e}\")"),

    # ---------- 9.1 ----------
    md("## 9. Численная оптимизация\n\n### Задание 9.1. Шаг и траектория"),
    code("learning_rates = [0.1, 1.0, 20.0]\n"
         "iterations = 100\n"
         "plt.figure(figsize=(8,4))\n"
         "for eta in learning_rates:\n"
         "    W = W_initial.copy(); b = b_initial.copy()\n"
         "    losses = [loss_and_grads(W, b)[0]]\n"
         "    for _ in range(iterations):\n"
         "        _, gW, gb = loss_and_grads(W, b)\n"
         "        W = W - eta*gW; b = b - eta*gb\n"
         "        losses.append(loss_and_grads(W, b)[0])\n"
         "    losses = np.array(losses)\n"
         "    print(f\"η={eta}: начальный={losses[0]:.4f}, конечный={losses[-1]:.4f}, \"\n"
         "          f\"min={losses.min():.4f} на шаге {losses.argmin()}\")\n"
         "    plt.semilogy(losses, label=f'η={eta}')\n"
         "plt.xlabel('итерация'); plt.ylabel('L'); plt.legend(); plt.grid(True)\n"
         "plt.show()"),

    # ---------- 10.1 ----------
    md("## 10. Итоговая карта понятий\n\n"
       "Предел задаёт производную как мгновенную скорость. Дифференциал "
       "даёт линейное приближение в окрестности точки. Частная производная "
       "меняет одну координату, а вектор всех частных производных — это "
       "градиент. Производная по направлению — скалярное произведение "
       "градиента на единичный вектор, а полная производная учитывает все "
       "пути влияния через зависимые аргументы. Якобиан собирает градиенты "
       "компонент векторного выхода по строкам; JVP отправляет вектор "
       "направления вперёд, VJP — ковектор чувствительности назад. Цепное "
       "правило превращает композицию в произведение якобианов. Гессиан — "
       "якобиан градиента, он описывает кривизну и позволяет отличить "
       "минимум от седла. След инвариантен относительно циклической "
       "перестановки, что удобно для матричных градиентов: "
       "$\\partial(\\operatorname{tr}(X^\\top A))/\\partial X = A$, "
       "$\\partial(a^\\top Xb)/\\partial X = ab^\\top$. В сквозной функции "
       "$Z=XW+\\mathbf 1b^\\top$ суммы по $d$ дают матричное произведение, "
       "сигмоида умножается поэлементно, а $\\nabla_W\\mathcal L=X^\\top G_Z+\\lambda W$ "
       "имеет ту же форму $3\\times2$, что и $W$. Конечные разности "
       "проверяют каждый элемент градиента, а численная оптимизация "
       "использует проверенные градиенты для уменьшения функции."),
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

with open("calculus_matrix_calculus_lab.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)

print("Готово: calculus_matrix_calculus_lab.ipynb")