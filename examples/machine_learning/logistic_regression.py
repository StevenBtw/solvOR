"""
Logistic Regression with Gradient-Based Optimizers

Binary classification using logistic regression, trained with different
gradient-based optimizers.

Formulation:
    minimize -sum(y*log(p) + (1-y)*log(1-p))  (cross-entropy loss)
    where p = sigmoid(X @ w + b)

Why these solvers:
    Logistic regression has a smooth, convex loss function ideal for
    gradient-based optimization. Adam typically converges fastest.

Expected result:
    All optimizers achieve similar classification accuracy (>90% on test data).
"""

from math import exp, log
from random import gauss, shuffle
from random import seed as set_seed

from solvor import adam, gradient_descent, momentum


def sigmoid(z):
    """Numerically stable sigmoid function."""
    if z >= 0:
        return 1 / (1 + exp(-z))
    else:
        ez = exp(z)
        return ez / (1 + ez)


def generate_data(n_samples=200, n_features=2, seed=42):
    """Generate synthetic binary classification data."""
    set_seed(seed)

    data = []
    # Class 0: centered around (-1, -1)
    for _ in range(n_samples // 2):
        x = [gauss(-1, 0.8) for _ in range(n_features)]
        data.append((x, 0))

    # Class 1: centered around (1, 1)
    for _ in range(n_samples // 2):
        x = [gauss(1, 0.8) for _ in range(n_features)]
        data.append((x, 1))

    shuffle(data)
    X = [d[0] for d in data]
    y = [d[1] for d in data]
    return X, y


def cross_entropy_loss(params, X, y, reg=0.01):
    """Cross-entropy loss with L2 regularization."""
    n_features = len(X[0])
    w = params[:n_features]
    b = params[n_features]
    n = len(y)

    loss = 0.0
    for xi, yi in zip(X, y):
        z = sum(wj * xj for wj, xj in zip(w, xi)) + b
        p = sigmoid(z)
        # Clip to avoid log(0)
        p = max(1e-15, min(1 - 1e-15, p))
        loss -= yi * log(p) + (1 - yi) * log(1 - p)

    # L2 regularization
    loss += reg * sum(wj * wj for wj in w)
    return loss / n


def cross_entropy_gradient(params, X, y, reg=0.01):
    """Gradient of cross-entropy loss."""
    n_features = len(X[0])
    w = params[:n_features]
    b = params[n_features]
    n = len(y)

    grad_w = [0.0] * n_features
    grad_b = 0.0

    for xi, yi in zip(X, y):
        z = sum(wj * xj for wj, xj in zip(w, xi)) + b
        p = sigmoid(z)
        error = (p - yi) / n

        for j in range(n_features):
            grad_w[j] += error * xi[j]
        grad_b += error

    # L2 regularization gradient
    for j in range(n_features):
        grad_w[j] += 2 * reg * w[j] / n

    return tuple(grad_w) + (grad_b,)


def predict(params, X):
    """Predict class labels."""
    n_features = len(X[0])
    w = params[:n_features]
    b = params[n_features]

    predictions = []
    for xi in X:
        z = sum(wj * xj for wj, xj in zip(w, xi)) + b
        p = sigmoid(z)
        predictions.append(1 if p >= 0.5 else 0)
    return predictions


def accuracy(y_pred, y_true):
    """Calculate classification accuracy."""
    correct = sum(1 for p, t in zip(y_pred, y_true) if p == t)
    return correct / len(y_true)


def main():
    # Generate data
    X, y = generate_data(n_samples=300, n_features=2)
    n_features = len(X[0])

    # Split into train/test
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print("Logistic Regression - Optimizer Comparison")
    print("=" * 55)
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    print()

    # Initial parameters (zeros)
    initial = tuple([0.0] * (n_features + 1))

    # Gradient function (closure over training data)
    def gradient(params):
        return cross_entropy_gradient(params, X_train, y_train)

    # Gradient Descent
    result_gd = gradient_descent(gradient, initial, lr=1.0, max_iter=500)
    acc_gd = accuracy(predict(result_gd.solution, X_test), y_test)

    # Momentum
    result_mom = momentum(gradient, initial, lr=1.0, beta=0.9, max_iter=500)
    acc_mom = accuracy(predict(result_mom.solution, X_test), y_test)

    # Adam
    result_adam = adam(gradient, initial, lr=0.1, max_iter=500)
    acc_adam = accuracy(predict(result_adam.solution, X_test), y_test)

    # Results
    print("Results:")
    print("-" * 55)
    print(f"{'Optimizer':<12} {'Loss':<12} {'Iters':<10} {'Test Acc':<10}")
    print("-" * 55)
    print(f"{'Gradient':<12} {result_gd.objective:<12.6f} {result_gd.iterations:<10} {acc_gd:<10.1%}")
    print(f"{'Momentum':<12} {result_mom.objective:<12.6f} {result_mom.iterations:<10} {acc_mom:<10.1%}")
    print(f"{'Adam':<12} {result_adam.objective:<12.6f} {result_adam.iterations:<10} {acc_adam:<10.1%}")

    # Show learned weights
    print()
    w = list(result_adam.solution[:n_features])
    b = result_adam.solution[n_features]
    print(f"Adam weights: [{', '.join(f'{wi:.3f}' for wi in w)}], bias: {b:.3f}")


if __name__ == "__main__":
    main()
