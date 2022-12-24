import numpy as np

import util
from linear_model import LinearModel


class LogisticRegression(LinearModel):
    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """Run Newton's method to minimize J(theta) for logistic regression

        Args:
            x (np.ndarray): Training example inputs. shape (m, n)
            y (np.ndarray): Training example labels (The thing we're predicting). shape (m,)
        """

        # Initialize theta based on number of features
        m, n = x.shape
        self.theta = np.zeros(n)

        # Newton's method
        while True:
            # Save old theta to compare later
            old_theta = np.copy(self.theta)

            # Compute Hessian Matrix
            h_x = 1 / (1 + np.exp(-x.dot(self.theta)))
            H = (x.T * h_x * (1 - h_x)).dot(x) / m
            gradient_J_theta = x.T.dot(h_x - y) // m

            # Update Theta
            self.theta -= np.linalg.inv(H).dot(gradient_J_theta)

            # End training if norm of theta_k - theta_k-1 < epsilon
            if np.linalg.norm(self.theta - old_theta, ord=1) < self.eps:
                break

    def predict(self, x: np.ndarray) -> list[int]:
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m,n).

        Returns:
            Outputs of shape (m,).
        """
        y_preds: list[int] = []
        for elem in x[0]:
            y = round(util.sigmoid(self.theta.dot(elem)))
            y_preds.append(y)

        return y_preds
