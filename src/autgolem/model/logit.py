import torch


class LogisticRegression:
    """Binary logistic regression trained with cross-entropy and mini-batch SGD in PyTorch."""

    def __init__(self, lr=0.1, epochs=20, batch_size=64):
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.loss_history = []

    def sigmoid(self, z):
        return 1 / (1 + torch.exp(-z))

    def cross_entropy(self, p, y):
        p = torch.clamp(p, 1e-7, 1 - 1e-7)        # avoid log(0)
        return -torch.mean(y * torch.log(p) + (1 - y) * torch.log(1 - p))

    def fit(self, X, y):
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        # standardize features
        self.mean, self.std = X.mean(0), X.std(0)
        X = (X - self.mean) / self.std

        self.w = torch.zeros(X.shape[1], requires_grad=True)
        self.b = torch.zeros(1, requires_grad=True)

        n = len(y)
        for epoch in range(self.epochs):
            perm = torch.randperm(n)                      # shuffle each epoch
            for i in range(0, n, self.batch_size):        # stochastic gradient descent (mini-batches)
                idx = perm[i:i + self.batch_size]
                p = self.sigmoid(X[idx] @ self.w + self.b)
                loss = self.cross_entropy(p, y[idx])
                loss.backward()
                with torch.no_grad():
                    self.w -= self.lr * self.w.grad
                    self.b -= self.lr * self.b.grad
                    self.w.grad.zero_()
                    self.b.grad.zero_()
            # cross-entropy on the full training set after each epoch
            with torch.no_grad():
                full_loss = self.cross_entropy(self.sigmoid(X @ self.w + self.b), y)
            self.loss_history.append(full_loss.item())
        return self

    def predict_proba(self, X):
        X = (torch.tensor(X, dtype=torch.float32) - self.mean) / self.std
        with torch.no_grad():
            return self.sigmoid(X @ self.w + self.b).numpy()

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)