"""
Section 1.2 Computer Problems - Fixed-Point Iteration (FPI)
=============================================================

Solves problems 1-5 using a generic Fixed-Point Iteration routine.

For an equation to be solved, we rewrite it in the form  x = g(x)
and iterate  x_{n+1} = g(x_n)  starting from an initial guess x0,
until two successive iterates agree to (at least) 8 correct decimal
places (or the requested tolerance).

Convergence theory used for the discussion in Problem 5:
Theorem 1.6 (Fixed-Point Iteration Theorem, simplified):
    If g is continuously differentiable near a fixed point r
    (i.e. g(r) = r) and |g'(r)| < 1, then FPI converges to r for
    any initial guess close enough to r (this is called "attracting"
    or "locally convergent"). If |g'(r)| > 1, the fixed point is
    "repelling" and FPI will NOT converge to it (except by luck,
    landing exactly on it).
"""

import math


def fixed_point_iteration(g, x0, tol=0.5e-8, max_iter=200, verbose=True, label="",
                           print_every=1):
    """
    Generic Fixed-Point Iteration.

    Parameters
    ----------
    g        : function, the iteration function x_{n+1} = g(x_n)
    x0       : float, initial guess
    tol      : float, stop when |x_{n+1} - x_n| < tol
    max_iter : int, maximum number of iterations allowed
    verbose  : bool, print the iteration history
    label    : str, name of the problem (for printing)

    Returns
    -------
    x        : float, the approximate fixed point
    n        : int, number of iterations used
    """
    x_old = x0
    history = [x_old]

    for n in range(1, max_iter + 1):
        x_new = g(x_old)
        history.append(x_new)

        if verbose and (n % print_every == 0):
            print(f"  n={n:6d}   x_{n} = {x_new:.10f}")

        if abs(x_new - x_old) < tol:
            if verbose:
                print(f"  --> Converged in {n} steps.")
                print(f"  --> Fixed point r ~= {x_new:.8f}\n")
            return x_new, n

        x_old = x_new

    print(f"  !! Did not converge within {max_iter} iterations for {label}.\n")
    return x_old, max_iter


def print_header(title):
    print("=" * 70)
    print(title)
    print("=" * 70)


# ---------------------------------------------------------------------
# Problem 1: FPI to 8 correct decimal places
# ---------------------------------------------------------------------
def problem1():
    print_header("PROBLEM 1: Fixed-Point Iteration (8 decimal places)")

    # (a) x^3 = 2x + 2  ->  x = (2x + 2)^(1/3)
    print("\n(a) x^3 = 2x + 2   ->   g(x) = (2x + 2)^(1/3)")
    g_a = lambda x: (2 * x + 2) ** (1 / 3)
    fixed_point_iteration(g_a, x0=1.0, label="1(a)")

    # (b) e^x + x = 7  ->  e^x = 7 - x  ->  x = ln(7 - x)
    print("(b) e^x + x = 7   ->   g(x) = ln(7 - x)")
    g_b = lambda x: math.log(7 - x)
    fixed_point_iteration(g_b, x0=1.0, label="1(b)")

    # (c) e^x + sin x = 4  ->  x = ln(4 - sin x)
    print("(c) e^x + sin(x) = 4   ->   g(x) = ln(4 - sin x)")
    g_c = lambda x: math.log(4 - math.sin(x))
    fixed_point_iteration(g_c, x0=1.0, label="1(c)")


# ---------------------------------------------------------------------
# Problem 2: FPI to 8 correct decimal places
# ---------------------------------------------------------------------
def problem2():
    print_header("PROBLEM 2: Fixed-Point Iteration (8 decimal places)")

    # (a) x^5 + x = 1  ->  x = (1 - x)^(1/5)
    print("\n(a) x^5 + x = 1   ->   g(x) = (1 - x)^(1/5)")
    g_a = lambda x: (1 - x) ** (1 / 5)
    fixed_point_iteration(g_a, x0=0.5, label="2(a)")

    # (b) sin x = 6x + 5  ->  x = (sin x - 5) / 6
    print("(b) sin(x) = 6x + 5   ->   g(x) = (sin x - 5) / 6")
    g_b = lambda x: (math.sin(x) - 5) / 6
    fixed_point_iteration(g_b, x0=-1.0, label="2(b)")

    # (c) ln x + x^2 = 3  ->  x = sqrt(3 - ln x)
    print("(c) ln(x) + x^2 = 3   ->   g(x) = sqrt(3 - ln x)")
    g_c = lambda x: math.sqrt(3 - math.log(x))
    fixed_point_iteration(g_c, x0=1.5, label="2(c)")


# ---------------------------------------------------------------------
# Problem 3: Square roots via FPI  (Example 1.6 style: g(x) = (x + A/x)/2)
# ---------------------------------------------------------------------
def problem3():
    print_header("PROBLEM 3: Square Roots via Fixed-Point Iteration")
    print("g(x) = (x + A/x) / 2   (averaging iteration, as in Example 1.6)\n")

    for A, x0 in [(3, 1.0), (5, 1.0)]:
        print(f"sqrt({A}):  initial guess x0 = {x0}")
        g = lambda x, A=A: (x + A / x) / 2
        fixed_point_iteration(g, x0=x0, label=f"sqrt({A})")


# ---------------------------------------------------------------------
# Problem 4: Cube roots via FPI, g(x) = (2x + A/x^2) / 3
# ---------------------------------------------------------------------
def problem4():
    print_header("PROBLEM 4: Cube Roots via Fixed-Point Iteration")
    print("g(x) = (2x + A / x^2) / 3\n")

    for A, x0 in [(2, 1.0), (3, 1.0), (5, 1.0)]:
        print(f"cube root of {A}:  initial guess x0 = {x0}")
        g = lambda x, A=A: (2 * x + A / x ** 2) / 3
        fixed_point_iteration(g, x0=x0, label=f"cbrt({A})")


# ---------------------------------------------------------------------
# Problem 5: g(x) = cos^2(x) -- convergence, fixed point, local analysis
# ---------------------------------------------------------------------
def problem5():
    print_header("PROBLEM 5: g(x) = cos^2(x)  (compare to Example 1.3, g(x)=cos x)")

    g = lambda x: math.cos(x) ** 2
    # Fixed point to 6 correct decimal places -> tighter tolerance.
    # Note: convergence here is much SLOWER than g(x)=cos(x) in Example 1.3,
    # because |g'(r)| turns out to be close to 1 (see analysis below), so we
    # allow many more iterations and only print a sample of them.
    print("\nRunning FPI for g(x) = cos^2(x), x0 = 1.0, tol = 0.5e-6:")
    r, n = fixed_point_iteration(g, x0=1.0, tol=0.5e-6, max_iter=200000,
                                  verbose=True, print_every=5000, label="5")

    print(f"Fixed point found: r ~= {r:.6f}")
    print(f"Number of FPI steps needed: {n}\n")

    # Local convergence analysis using Theorem 1.6:
    # g'(x) = 2 cos(x) * (-sin(x)) = -sin(2x)
    g_prime_r = -math.sin(2 * r)
    print("Local convergence discussion (Theorem 1.6):")
    print(f"  g'(x) = d/dx[cos^2(x)] = -sin(2x)")
    print(f"  g'(r) = -sin(2 * {r:.6f}) = {g_prime_r:.6f}")
    if abs(g_prime_r) < 1:
        print(f"  |g'(r)| = {abs(g_prime_r):.6f} < 1  =>  r is an ATTRACTING fixed point.")
        print("  So, like g(x) = cos(x) in Example 1.3, FPI with g(x) = cos^2(x)")
        print("  also converges locally near r (though the convergence here is")
        print("  slower/faster depending on how close |g'(r)| is to 0 or 1).")
    else:
        print(f"  |g'(r)| = {abs(g_prime_r):.6f} >= 1  =>  r is a REPELLING fixed point,")
        print("  so FPI would NOT converge to r from nearby guesses.")


# ---------------------------------------------------------------------
if __name__ == "__main__":
    problem1()
    problem2()
    problem3()
    problem4()
    problem5()