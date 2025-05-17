"""
Hyperbolic Fuzzy C-Means (HypeFCM) Clustering
Reference: "Hyperbolic Fuzzy C-Means with Adaptive Weight-based Filtering for Clustering in Non-Euclidean Spaces"

A unified script for clustering in non-Euclidean spaces using the Poincaré ball model.
Run directly as a .py script or in a Jupyter notebook.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.manifold import TSNE

def mobius_add(x, y):
    """Möbius addition in the Poincaré disc model."""
    x2 = np.sum(x ** 2, axis=-1, keepdims=True)
    y2 = np.sum(y ** 2, axis=-1, keepdims=True)
    xy = np.sum(x * y, axis=-1, keepdims=True)
    numerator = (1 + 2 * xy + y2) * x + (1 - x2) * y
    denominator = 1 + 2 * xy + x2 * y2
    return numerator / denominator

def log_map(x, y, eps=1e-8):
    """Logarithmic map in the Poincaré disc model."""
    mobius_diff = mobius_add(-x, y)
    norm = np.linalg.norm(mobius_diff, axis=-1, keepdims=True)
    norm = np.maximum(norm, eps)  # prevent division by zero
    lambda_x = 2 / (1 - np.sum(x ** 2, axis=-1, keepdims=True))
    return (2 / lambda_x) * np.arctanh(norm) * (mobius_diff / norm)

def exp_map(x, v, eps=1e-8):
    """Exponential map in the Poincaré disc model."""
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    norm = np.maximum(norm, eps)
    lambda_x = 2 / (1 - np.sum(x ** 2, axis=-1, keepdims=True))
    second_term = np.tanh(lambda_x * norm / 2) * (v / norm)
    return mobius_add(x, second_term)

def poincare_distance(x, y, eps=1e-8):
    """Compute hyperbolic distance between two points in the Poincaré disc."""
    diff = mobius_add(-x, y)
    norm = np.linalg.norm(diff, axis=-1)
    norm = np.clip(norm, 0, 1 - eps)  # avoid arctanh(1)
    return 2 * np.arctanh(norm)

def initialize_membership(n, c):
    """Initialize fuzzy membership matrix."""
    return np.random.dirichlet(np.ones(c), size=n)

class HypeFCM:
    """
    Hyperbolic Fuzzy C-Means clustering with adaptive weight-based filtering.
    
    Args:
        n_clusters (int): Number of clusters (default=3).
        m (float): Fuzziness parameter (default=2.0).
        max_iter (int): Maximum iterations (default=100).
        tol (float): Convergence threshold (default=1e-5).
        filtration_k (int): Retain top-k closest points per centroid (default=5).
    """
    def __init__(self, n_clusters=3, m=2.0, max_iter=100, tol=1e-5, filtration_k=5):
        self.n_clusters = n_clusters
        self.m = m
        self.max_iter = max_iter
        self.tol = tol
        self.filtration_k = filtration_k
        self.centroids = None
        self.membership = None
        
    def _update_centroids(self, X, W, V_prev):
        """Update centroids using exponential and logarithmic maps."""
        V = []
        for j in range(self.n_clusters):
            weights = W[:, j] ** self.m
            tangent_vectors = np.array([
                log_map(V_prev[j], X[i]) * weights[i] for i in range(len(X))
            ])
            mean_vec = np.sum(tangent_vectors, axis=0) / np.sum(weights)
            new_centroid = exp_map(V_prev[j], mean_vec)
            V.append(new_centroid)
        return np.stack(V)
    
    def _update_membership(self, X, V):
        """Update membership weights with optional filtration."""
        n = len(X)
        U = np.array([[poincare_distance(X[i], V[j]) ** 2 for j in range(self.n_clusters)] for i in range(n)])
        
        if self.filtration_k is not None:
            top_k = np.argsort(U, axis=1)[:, :self.filtration_k]
            mask = np.zeros_like(U)
            for i in range(n):
                mask[i, top_k[i]] = 1
            U = np.where(mask, U, np.inf)

        U_inv = np.where(U == np.inf, 0, U ** (-1 / (self.m - 1)))
        row_sums = U_inv.sum(axis=1, keepdims=True)
        W = np.divide(U_inv, row_sums, where=row_sums != 0)
        return W
    
    def fit(self, X):
        """Fit HypeFCM to the data."""
        # Normalize to Poincaré ball
        norm_X = np.clip(np.linalg.norm(X, axis=1, keepdims=True), a_min=1e-5, a_max=None)
        X = 0.9 * X / norm_X  # embed in the unit ball (radius < 1)
        
        n = X.shape[0]
        self.membership = initialize_membership(n, self.n_clusters)
        self.centroids = X[np.random.choice(n, self.n_clusters, replace=False)]  # random init
        
        for _ in range(self.max_iter):
            V_new = self._update_centroids(X, self.membership, self.centroids)
            W_new = self._update_membership(X, V_new)
            
            if np.linalg.norm(W_new - self.membership) < self.tol:
                break
                
            self.centroids, self.membership = V_new, W_new
        
        return self

    def predict(self):
        """Return cluster labels."""
        return np.argmax(self.membership, axis=1)

    def evaluate(self, true_labels):
        """Calculate ARI and NMI scores."""
        pred_labels = self.predict()
        return {
            "ARI": adjusted_rand_score(true_labels, pred_labels),
            "NMI": normalized_mutual_info_score(true_labels, pred_labels)
        }

    def visualize(self, X, dim=2):
        """Visualize clusters (2D/3D)."""
        if dim not in [2, 3]:
            raise ValueError("Visualization supports 2D or 3D only.")
        
        # Reduce dimensionality if needed
        if X.shape[1] > dim:
            X_vis = TSNE(n_components=dim).fit_transform(X)
        else:
            X_vis = X
        
        plt.figure(figsize=(8, 6))
        if dim == 2:
            plt.scatter(X_vis[:, 0], X_vis[:, 1], c=self.predict(), cmap='viridis', alpha=0.6)
        elif dim == 3:
            ax = plt.axes(projection='3d')
            ax.scatter(X_vis[:, 0], X_vis[:, 1], X_vis[:, 2], c=self.predict(), cmap='viridis', alpha=0.6)
        plt.title("HypeFCM Clustering Result")
        plt.show()

# Example usage (works in .py and .ipynb)
if __name__ == "__main__":
    # Load data
    data = np.genfromtxt("wine.csv", delimiter=",", skip_header=1)
    X = data[:, :-1]
    true_labels = data[:, -1].astype(int)
    
    # Run HypeFCM
    model = HypeFCM(n_clusters=3, filtration_k=5)
    model.fit(X)
    
    # Evaluate
    scores = model.evaluate(true_labels)
    print(f"ARI: {scores['ARI']:.3f}, NMI: {scores['NMI']:.3f}")
    
    # Visualize
    model.visualize(X, dim=2)
