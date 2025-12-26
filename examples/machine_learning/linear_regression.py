"""
Linear Regression with Gradient-Based Optimizers

Train a linear regression model using different gradient-based optimizers
and compare their convergence.

Formulation:
    minimize (1/n) * sum((y_pred - y_true)^2)
    where y_pred = X @ w + b

Why these solvers:
    Gradient descent is the foundation, momentum adds acceleration,
    and Adam adapts learning rates per-parameter for faster convergence.

Expected result:
    All optimizers converge to similar weights with MSE near the noise level.
    Adam typically converges fastest, momentum second, gradient descent slowest.
"""

from solvor import adam, gradient_descent, momentum


def generate_data(n_samples=100, n_features=3, noise=0.1, seed=42):
    """Generate synthetic regression data."""
    from random import gauss, random
    from random import seed as set_seed

    set_seed(seed)

    # True weights
    true_w = [1.5, -2.0, 0.5][:n_features]
    true_b = 1.0

    # Generate X and y
    X = [[random() * 2 - 1 for _ in range(n_features)] for _ in range(n_samples)]
    y = []
    for row in X:
        pred = sum(w * x for w, x in zip(true_w, row)) + true_b
        y.append(pred + gauss(0, noise))

    return X, y, true_w, true_b


def mse_loss(params, X, y):
    """Mean squared error loss function."""
    n_features = len(X[0])
    w = params[:n_features]
    b = params[n_features]

    total = 0.0
    for xi, yi in zip(X, y):
        pred = sum(wj * xj for wj, xj in zip(w, xi)) + b
        total += (pred - yi) ** 2
    return total / len(y)


def mse_gradient(params, X, y):
    """Gradient of MSE loss."""
    n_features = len(X[0])
    n = len(y)
    w = params[:n_features]
    b = params[n_features]

    grad_w = [0.0] * n_features
    grad_b = 0.0

    for xi, yi in zip(X, y):
        pred = sum(wj * xj for wj, xj in zip(w, xi)) + b
        error = 2 * (pred - yi) / n
        for j in range(n_features):
            grad_w[j] += error * xi[j]
        grad_b += error

    return tuple(grad_w) + (grad_b,)


def main():
    # Generate data
    X, y, true_w, true_b = generate_data(n_samples=200, n_features=3)
    n_features = len(X[0])

    print("Linear Regression - Optimizer Comparison")
    print("=" * 50)
    print(f"True weights: {true_w}, bias: {true_b}")
    print()

    # Initial parameters (zeros)
    initial = tuple([0.0] * (n_features + 1))

    # Objective and gradient functions (closure over X, y)
    def objective(params):
        return mse_loss(params, X, y)

    def gradient(params):
        return mse_gradient(params, X, y)

    # Gradient Descent
    result_gd = gradient_descent(gradient, initial, lr=0.5, max_iter=1000)
    w_gd = list(result_gd.solution[:n_features])
    b_gd = result_gd.solution[n_features]

    # Momentum
    result_mom = momentum(gradient, initial, lr=0.5, beta=0.9, max_iter=1000)
    w_mom = list(result_mom.solution[:n_features])
    b_mom = result_mom.solution[n_features]

    # Adam
    result_adam = adam(gradient, initial, lr=0.1, max_iter=1000)
    w_adam = list(result_adam.solution[:n_features])
    b_adam = result_adam.solution[n_features]

    # Results
    print("Results:")
    print("-" * 50)
    print(f"{'Optimizer':<15} {'MSE':<12} {'Iterations':<12} Weights")
    print("-" * 50)

    print(f"{'Gradient':<15} {result_gd.objective:<12.6f} {result_gd.iterations:<12}", end="")
    print(f"[{', '.join(f'{w:.3f}' for w in w_gd)}], b={b_gd:.3f}")

    print(f"{'Momentum':<15} {result_mom.objective:<12.6f} {result_mom.iterations:<12}", end="")
    print(f"[{', '.join(f'{w:.3f}' for w in w_mom)}], b={b_mom:.3f}")

    print(f"{'Adam':<15} {result_adam.objective:<12.6f} {result_adam.iterations:<12}", end="")
    print(f"[{', '.join(f'{w:.3f}' for w in w_adam)}], b={b_adam:.3f}")

    print()
    print(f"True weights: [{', '.join(f'{w:.3f}' for w in true_w)}], b={true_b:.3f}")


if __name__ == "__main__":
    main()
