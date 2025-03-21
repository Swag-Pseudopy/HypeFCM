"""
Hyperbolic Fuzzy C-Means (HypeFCM) Clustering
Reference: "Hyperbolic Fuzzy C-Means with Adaptive Weight-Based Filtering"

A unified script for clustering in non-Euclidean spaces using the Poincaré ball model.
Run directly as a .py script or in a Jupyter notebook.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.manifold import TSNE

# Functions for hyperbolic operations
def mob_add(a, b, c=1.0):
    """Möbius addition in the Poincaré disc model."""
    numerator = (1 + 2 * c * np.dot(a, b) + c * np.dot(b, b)) * a + (1 - c * np.dot(a, a)) * b
    denominator = 1 + 2 * c * np.dot(a, b) + c**2 * np.dot(a, a) * np.dot(b, b)
    return numerator / denominator

def hyperbolic_dist(a, b, c=1.0):
    """Compute hyperbolic distance between two points in the Poincaré disc."""
    eps = 1e-9
    mob_diff = mob_add(-a, b, c)
    norm = np.clip((c * np.sum(mob_diff * mob_diff)) ** 0.5, -1 + eps, 1 - eps)
    return (2 / np.sqrt(c)) * np.arctanh(norm)

def mobius_scalar_mul(r, a, c=1.0):
    """Möbius scalar multiplication in the Poincaré disc model."""
    eps = 1e-9
    norm = np.clip((np.sum(a * a) * c) ** 0.5, -1 + eps, 1 - eps)
    return np.tanh(r * np.arctanh(norm)) * (a / norm)

class HypeFCM:
    """
    Hyperbolic Fuzzy C-Means clustering with adaptive weight-based filtering.
    
    Args:
        n_clusters (int): Number of clusters (default=3).
        m (float): Fuzziness parameter (default=2.0).
        curvature (float): Curvature of hyperbolic space (default=1.0).
        filtration_k (int): Retain top-k closest points per centroid (default=5).
        max_iter (int): Maximum iterations (default=100).
        tol (float): Convergence threshold (default=1e-5).
    """
    def __init__(self, n_clusters=3, m=2.0, curvature=1.0, filtration_k=5, max_iter=100, tol=1e-5):
        self.n_clusters = n_clusters
        self.m = m
        self.curvature = curvature
        self.filtration_k = filtration_k
        self.max_iter = max_iter
        self.tol = tol
        self.centroids = None
        self.membership = None

    def fit(self, X):
        """Fit HypeFCM to the data."""
        n_samples, n_features = X.shape
        self.membership = np.random.dirichlet(np.ones(self.n_clusters), n_samples)
        
        for _ in range(self.max_iter):
            # Update centroids using Möbius operations
            temp = np.zeros_like(X)
            for j in range(self.n_clusters):
                for i in range(n_samples):
                    temp[j] = mob_add(temp[j],
                                      mobius_scalar_mul((np.sum(self.membership[:, j] ** self.m)**(-1))*self.membership[i, j] ** self.m,
                                                        X[i],self.curvature),self.curvature)
            self.centroids = np.array(temp)
            
            # Compute distances and apply filtration
            distances = np.array([[hyperbolic_dist(X[i], self.centroids[j], self.curvature)
                                  for j in range(self.n_clusters)] for i in range(n_samples)])
            sorted_indices = np.argsort(distances, axis=1)
            mask = np.zeros_like(distances, dtype=bool)
            mask[np.arange(n_samples)[:, None], sorted_indices[:, :self.filtration_k]] = True
            filtered_distances = np.where(mask, distances, 0) + 1e-10
            
            # Update membership weights
            inv_dist = 1.0 / filtered_distances
            new_membership = inv_dist / np.sum(inv_dist, axis=1, keepdims=True)
            
            # Check convergence
            if np.linalg.norm(new_membership - self.membership) < self.tol:
                break
            self.membership = new_membership
        
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
            #plt.scatter(self.centroids[:, 0], self.centroids[:, 1], c='red', marker='X', s=200)
        elif dim == 3:
            ax = plt.axes(projection='3d')
            ax.scatter(X_vis[:, 0], X_vis[:, 1], X_vis[:, 2], c=self.predict(), cmap='viridis', alpha=0.6)
            #ax.scatter(self.centroids[:, 0], self.centroids[:, 1], self.centroids[:, 2], c='red', marker='X', s=200)
        plt.title("HypeFCM Clustering Result")
        plt.show()

# Example usage (works in .py and .ipynb)
if __name__ == "__main__":
    # Load data
    eps = 1e-9   
    data = np.genfromtxt("wine.csv", delimiter=",", skip_header=1)
    X = data[:, :-1]
    norm = np.linalg.norm(X,axis = 0)
    X = X/max(norm+eps)
    true_labels = data[:, -1].astype(int)
    
    # Run HypeFCM
    model = HypeFCM(n_clusters=3, curvature=1.0, filtration_k=5)
    model.fit(X)
    
    # Evaluate
    scores = model.evaluate(true_labels)
    print(f"ARI: {scores['ARI']:.3f}, NMI: {scores['NMI']:.3f}")
    
    # Visualize
    model.visualize(X, dim=2)
