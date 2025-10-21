import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson

def exercise4(num_points=75):
    """Generate and plot 75 uniformly random points in [-5,5] x [-5,5]."""
    points = np.random.uniform(-5, 5, size=(num_points, 2))

    plt.figure(figsize=(5, 5))
    plt.scatter(points[:, 0], points[:, 1], c='blue', alpha=0.7)
    plt.title(f"Exercise 4: {num_points} Uniform Random Points in [-5, 5] x [-5, 5]")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.grid(True)
    plt.axis("equal")
    plt.show()


def exercise5(mean_points=75):
    """Generate and plot L ~ Poisson(mean_points) uniformly random points in [-5,5] x [-5,5]."""
    L = np.random.poisson(mean_points)
    points = np.random.uniform(-5, 5, size=(L, 2))
    
    # --- Plot Poisson distribution ---
    plot_poisson_distribution(mean_points, L)

    plt.figure(figsize=(5, 5))
    plt.scatter(points[:, 0], points[:, 1], c='red', alpha=0.7)
    plt.title(f"Exercise 5: Poisson({mean_points}) = {L} Points in [-5, 5] x [-5, 5]")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.grid(True)
    plt.axis("equal")
    plt.show()

def plot_poisson_distribution(mean_points, sample_L=None):
    """Plot the Poisson distribution PMF for the given mean."""
    # Choose range around the mean
    k = np.arange(max(0, mean_points - 30), mean_points + 40)
    pmf = poisson.pmf(k, mean_points)

    plt.figure(figsize=(7, 4))
    plt.bar(k, pmf, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title(f"Poisson Distribution (λ = {mean_points})")
    plt.xlabel("Number of Points (L)")
    plt.ylabel("Probability")

    # Mark the sampled L if provided
    if sample_L is not None:
        plt.axvline(sample_L, color='red', linestyle='--', linewidth=2, label=f"Sampled L = {sample_L}")
        plt.legend()

    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()


if __name__ == "__main__":
    exercise4()
    exercise5()

