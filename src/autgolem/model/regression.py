import numpy as np
import torch
import matplotlib.pyplot as plt


class LinearRegression:
    def __init__(self, learning_rate=0.1, epochs=500):
        self.learning_rate = learning_rate
        self.epochs = epochs
        # parameters of y = w0 + w1 * x
        self.w0 = torch.zeros(1, requires_grad=True)
        self.w1 = torch.zeros(1, requires_grad=True)
        self.optimizer = torch.optim.SGD([self.w0, self.w1], lr=learning_rate)
        self.loss_fn = torch.nn.MSELoss()
        # training history
        self.w0_history = []
        self.w1_history = []
        self.loss_history = []

    def forward(self, x):
        return self.w0 + self.w1 * x

    def _to_tensor(self, a):
        return torch.tensor(np.asarray(a, dtype=np.float32)).flatten()

    def _original_weights(self):
        # training runs on standardized data -> convert back to original units
        w1 = self.w1.item() * self.y_std / self.x_std
        w0 = self.y_mean + self.y_std * self.w0.item() - w1 * self.x_mean
        return w0, w1

    def fit(self, x_train, y_train, x_test, y_test):
        x = self._to_tensor(x_train)
        y = self._to_tensor(y_train)

        # standardize, otherwise SGD diverges (AnnualProduction is in the thousands)
        self.x_mean, self.x_std = x.mean().item(), x.std().item()
        self.y_mean, self.y_std = y.mean().item(), y.std().item()
        xs = (x - self.x_mean) / self.x_std
        ys = (y - self.y_mean) / self.y_std

        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            loss = self.loss_fn(self.forward(xs), ys)
            loss.backward()
            self.optimizer.step()

            w0, w1 = self._original_weights()
            self.w0_history.append(w0)
            self.w1_history.append(w1)
            self.loss_history.append(loss.item())

        self.x_train, self.y_train = x, y

        # R^2 on test data
        y_true = self._to_tensor(y_test)
        y_pred = self.predict(x_test)
        ss_res = ((y_true - y_pred) ** 2).sum()
        ss_tot = ((y_true - y_true.mean()) ** 2).sum()
        self.r2 = (1 - ss_res / ss_tot).item()
        print(f"R^2 on test data: {self.r2:.4f}")
        return self.r2

    def predict(self, x):
        x = self._to_tensor(x)
        w0, w1 = self._original_weights()
        return w0 + w1 * x

    def analysis_plot(self):
        fig, ax = plt.subplots(2, 2, figsize=(12, 8))

        x_line = torch.linspace(self.x_train.min().item(), self.x_train.max().item(), 100)
        ax[0, 0].scatter(self.x_train, self.y_train, s=10, label="training data")
        ax[0, 0].plot(x_line, self.predict(x_line), color="red", label="fitted line")
        ax[0, 0].set_xlabel("BCR")
        ax[0, 0].set_ylabel("AnnualProduction")
        ax[0, 0].set_title("Data and fitted regression line")
        ax[0, 0].legend()

        ax[0, 1].plot(self.loss_history)
        ax[0, 1].set_title("Loss (MSE, standardized)")
        ax[0, 1].set_xlabel("Epoch")

        ax[1, 0].plot(self.w0_history)
        ax[1, 0].set_title("w0 during training")
        ax[1, 0].set_xlabel("Epoch")

        ax[1, 1].plot(self.w1_history)
        ax[1, 1].set_title("w1 during training")
        ax[1, 1].set_xlabel("Epoch")

        for a in ax.flat:
            a.grid()
        plt.tight_layout()
        plt.show()