# Anomaly Detection - Lab 2

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score

from pyod.utils.data import generate_data_clusters
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.utils.utility import standardizer
from pyod.models.combination import average, maximization

# EXERCISE 1: LEVERAGE SCORES

def calculate_leverage(X):
    """Calculate leverage scores for a linear regression model."""
    X_design = np.column_stack((np.ones(X.shape[0]), X))
    hat_matrix = (
        X_design
        @ np.linalg.pinv(X_design.T @ X_design)
        @ X_design.T
    )
    return np.diag(hat_matrix)


def exercise1():
    print("\nEXERCISE 1: LEVERAGE SCORES")

    rng = np.random.default_rng(42)
    n = 40

    # Each case specifies the noise mean (mu) and variance (sigma^2).
    noise_cases = [
        (0.0, 0.1),
        (1.0, 1.0),
        (2.0, 4.0),
    ]

    colors = {
        "Regular": "tab:blue",
        "High variance X": "tab:orange",
        "High variance Y": "tab:green",
        "High variance X and Y": "tab:red",
    }
 
    # 1D model: y = a*x + b + epsilon
  
    a, b = 2.0, 1.0

    fig1, axes1 = plt.subplots(1, 3, figsize=(17, 5))

    for ax, (mu, variance) in zip(axes1, noise_cases):
        groups = []

        # Regular observations
        x = rng.normal(0, 1, n)
        y = a * x + b + rng.normal(mu, np.sqrt(variance), n)
        groups.append(("Regular", x, y))

        # High variance in X
        x = rng.normal(0, 5, n)
        y = a * x + b + rng.normal(mu, np.sqrt(variance), n)
        groups.append(("High variance X", x, y))

        # High variance in Y
        x = rng.normal(0, 1, n)
        y = a * x + b + rng.normal(
            mu, 5 * np.sqrt(variance), n
        )
        groups.append(("High variance Y", x, y))

        # High variance in both X and Y
        x = rng.normal(0, 5, n)
        y = a * x + b + rng.normal(
            mu, 5 * np.sqrt(variance), n
        )
        groups.append(("High variance X and Y", x, y))

        x_all = np.concatenate([group[1] for group in groups])
        y_all = np.concatenate([group[2] for group in groups])

        leverage = calculate_leverage(x_all.reshape(-1, 1))
        top = np.argsort(leverage)[-5:]

        for name, x_group, y_group in groups:
            ax.scatter(
                x_group,
                y_group,
                color=colors[name],
                s=20,
                alpha=0.65,
                label=name,
            )

        # Circle the five observations with highest leverage
        ax.scatter(
            x_all[top],
            y_all[top],
            s=110,
            facecolors="none",
            edgecolors="black",
            linewidths=1.5,
            label="Top 5 leverage",
        )

        x_line = np.linspace(x_all.min(), x_all.max(), 100)
        ax.plot(
            x_line,
            a * x_line + b + mu,
            "k--",
            label="Mean model",
        )

        ax.set_title(f"1D: mu={mu}, variance={variance}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.grid(alpha=0.2)

    handles, labels = axes1[0].get_legend_handles_labels()
    fig1.legend(handles, labels, loc="lower center", ncol=3)
    fig1.tight_layout(rect=[0, 0.17, 1, 1])

    # 2D model: y = a*x1 + b*x2 + c + epsilon
   
    a, b, c = 2.0, -1.5, 5.0

    fig2, axes2 = plt.subplots(1, 3, figsize=(17, 5))

    for ax, (mu, variance) in zip(axes2, noise_cases):
        groups = []

        # Regular observations
        x1 = rng.normal(0, 1, n)
        x2 = rng.normal(0, 1, n)
        y = (
            a * x1 + b * x2 + c
            + rng.normal(mu, np.sqrt(variance), n)
        )
        groups.append(("Regular", x1, x2, y))

        # High variance in the input features
        x1 = rng.normal(0, 5, n)
        x2 = rng.normal(0, 5, n)
        y = (
            a * x1 + b * x2 + c
            + rng.normal(mu, np.sqrt(variance), n)
        )
        groups.append(("High variance X", x1, x2, y))

        # High variance in Y (noise)
        x1 = rng.normal(0, 1, n)
        x2 = rng.normal(0, 1, n)
        y = (
            a * x1 + b * x2 + c
            + rng.normal(mu, 5 * np.sqrt(variance), n)
        )
        groups.append(("High variance Y", x1, x2, y))

        # High variance in inputs and Y
        x1 = rng.normal(0, 5, n)
        x2 = rng.normal(0, 5, n)
        y = (
            a * x1 + b * x2 + c
            + rng.normal(mu, 5 * np.sqrt(variance), n)
        )
        groups.append(("High variance X and Y", x1, x2, y))

        x1_all = np.concatenate([group[1] for group in groups])
        x2_all = np.concatenate([group[2] for group in groups])

        X_all = np.column_stack((x1_all, x2_all))
        leverage = calculate_leverage(X_all)
        top = np.argsort(leverage)[-5:]

        for name, x1_group, x2_group, _ in groups:
            ax.scatter(
                x1_group,
                x2_group,
                color=colors[name],
                s=20,
                alpha=0.65,
                label=name,
            )

        ax.scatter(
            x1_all[top],
            x2_all[top],
            s=110,
            facecolors="none",
            edgecolors="black",
            linewidths=1.5,
            label="Top 5 leverage",
        )

        ax.set_title(f"2D: mu={mu}, variance={variance}")
        ax.set_xlabel("X1")
        ax.set_ylabel("X2")
        ax.grid(alpha=0.2)

    handles, labels = axes2[0].get_legend_handles_labels()
    fig2.legend(handles, labels, loc="lower center", ncol=3)
    fig2.tight_layout(rect=[0, 0.17, 1, 1])

def exercise2():
    print("\nEXERCISE 2: KNN ON SYNTHETIC DATA")

    X_train, X_test, y_train, y_test = generate_data_clusters(
        n_train=400,
        n_test=200,
        n_clusters=2,
        n_features=2,
        contamination=0.1,
        size="same",
        density="same",
        dist=0.25,
        random_state=42,
        return_in_clusters=False,
    )

    y_train = np.asarray(y_train).ravel().astype(int)
    y_test = np.asarray(y_test).ravel().astype(int)

    neighbor_values = [5, 10, 20, 30, 40]
    results = {}

    print("\nBalanced accuracy:")
    print(f"{'n_neighbors':>12} {'Train BA':>12} {'Test BA':>12}")

    for k in neighbor_values:
        model = KNN(n_neighbors=k, contamination=0.1)
        model.fit(X_train)

        train_pred = model.labels_
        test_pred = model.predict(X_test)

        train_ba = balanced_accuracy_score(y_train, train_pred)
        test_ba = balanced_accuracy_score(y_test, test_pred)

        results[k] = (train_pred, test_pred)

        print(f"{k:>12} {train_ba:>12.4f} {test_ba:>12.4f}")

    # Display the four requested panels for k=20.
    selected_k = 20
    train_pred, test_pred = results[selected_k]

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    panels = [
        (axes[0, 0], X_train, y_train, "Train: ground truth"),
        (
            axes[0, 1],
            X_train,
            train_pred,
            f"Train: KNN prediction (k={selected_k})",
        ),
        (axes[1, 0], X_test, y_test, "Test: ground truth"),
        (
            axes[1, 1],
            X_test,
            test_pred,
            f"Test: KNN prediction (k={selected_k})",
        ),
    ]

    for ax, X, labels, title in panels:
        labels = np.asarray(labels).ravel()
        inliers = labels == 0
        outliers = labels == 1

        ax.scatter(
            X[inliers, 0],
            X[inliers, 1],
            c="tab:blue",
            s=18,
            label="Inliers",
        )
        ax.scatter(
            X[outliers, 0],
            X[outliers, 1],
            c="tab:red",
            s=24,
            label="Outliers",
        )

        ax.set_title(title)
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        ax.legend()
        ax.grid(alpha=0.2)

    fig.tight_layout()

def exercise3():
    print("\nEXERCISE 3: KNN VERSUS LOF")

    X, _ = make_blobs(
        n_samples=[200, 100],
        n_features=2,
        centers=[(-10, -10), (10, 10)],
        cluster_std=[2, 6],
        shuffle=True,
        random_state=42,
    )

    contamination = 0.07
    neighbor_values = [5, 10, 20, 30, 40]
    results = {}

    print("\nPredicted outliers and disagreement:")
    print(
        f"{'Neighbors':>10} {'KNN outliers':>13} "
        f"{'LOF outliers':>13} {'Different labels':>17}"
    )

    for k in neighbor_values:
        knn_model = KNN(
            n_neighbors=k,
            contamination=contamination,
        )
        lof_model = LOF(
            n_neighbors=k,
            contamination=contamination,
        )

        knn_model.fit(X)
        lof_model.fit(X)

        knn_pred = knn_model.labels_
        lof_pred = lof_model.labels_

        results[k] = (knn_pred, lof_pred)

        knn_count = int(np.sum(knn_pred == 1))
        lof_count = int(np.sum(lof_pred == 1))
        different = int(np.sum(knn_pred != lof_pred))

        print(
            f"{k:>10} {knn_count:>13} "
            f"{lof_count:>13} {different:>17}"
        )

    # Show the two methods for k=20.
    selected_k = 20
    knn_pred, lof_pred = results[selected_k]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    panels = [
        (axes[0], knn_pred, f"KNN (k={selected_k})"),
        (axes[1], lof_pred, f"LOF (k={selected_k})"),
    ]

    for ax, labels, title in panels:
        inliers = labels == 0
        outliers = labels == 1

        ax.scatter(
            X[inliers, 0],
            X[inliers, 1],
            c="tab:blue",
            s=20,
            label="Inliers",
        )
        ax.scatter(
            X[outliers, 0],
            X[outliers, 1],
            c="tab:red",
            s=28,
            label="Predicted outliers",
        )

        ax.set_title(title)
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        ax.legend()
        ax.grid(alpha=0.2)

    fig.tight_layout()

def find_cardio_file():
    """Find cardio.mat in the lab2 directory or its subdirectories."""
    folder = Path(__file__).resolve().parent

    direct_path = folder / "cardio.mat"
    if direct_path.exists():
        return direct_path

    matches = list(folder.rglob("cardio.mat"))
    return matches[0] if matches else None


def load_cardio_dataset(file_path):
    """Load features and anomaly labels from cardio.mat."""
    from scipy.io import loadmat

    data = loadmat(str(file_path))

    X = None
    y = None

    for key in ("X", "x", "data", "features"):
        if key in data:
            candidate = np.asarray(data[key])
            if candidate.ndim == 2 and candidate.size > 0:
                X = candidate
                break

    for key in ("y", "Y", "label", "labels", "ground_truth"):
        if key in data:
            candidate = np.asarray(data[key]).ravel()
            if candidate.size > 0:
                y = candidate
                break

    if X is None or y is None:
        keys = [key for key in data if not key.startswith("__")]
        raise ValueError(
            f"Could not find X and y in cardio.mat. Variables: {keys}"
        )

    X = np.asarray(X, dtype=float)
    y = np.asarray(y).ravel()

    if X.shape[0] != len(y) and X.shape[1] == len(y):
        X = X.T

    if X.shape[0] != len(y):
        raise ValueError("Feature and label counts do not match.")

    # Convert labels to 0 = inlier and 1 = anomaly.
    unique = np.unique(y)

    if set(unique.tolist()) == {-1, 1}:
        y = (y == 1).astype(int)
    elif set(unique.tolist()) == {0, 1}:
        y = y.astype(int)
    else:
        values, counts = np.unique(y, return_counts=True)
        anomaly_label = values[np.argmin(counts)]
        print(f"Treating label {anomaly_label} as the anomaly class.")
        y = (y == anomaly_label).astype(int)

    # Remove rows with invalid feature values.
    valid = np.isfinite(X).all(axis=1)
    X = X[valid]
    y = y[valid]

    return X, y

def evaluate_ensemble(
    algorithm,
    X_train,
    X_test,
    y_train,
    y_test,
    contamination,
):
    """Evaluate ten models and combine their anomaly scores."""
    neighbor_values = np.linspace(30, 120, 10, dtype=int)

    train_scores = []
    test_scores = []

    print(f"\n{algorithm} models:")
    print(f"{'n_neighbors':>12} {'Train BA':>12} {'Test BA':>12}")

    for k in neighbor_values:
        if algorithm == "KNN":
            model = KNN(
                n_neighbors=int(k),
                contamination=contamination,
            )
        else:
            model = LOF(
                n_neighbors=int(k),
                contamination=contamination,
            )

        model.fit(X_train)

        train_pred = model.labels_
        test_pred = model.predict(X_test)

        train_ba = balanced_accuracy_score(y_train, train_pred)
        test_ba = balanced_accuracy_score(y_test, test_pred)

        print(f"{k:>12} {train_ba:>12.4f} {test_ba:>12.4f}")

        train_scores.append(model.decision_scores_)
        test_scores.append(model.decision_function(X_test))

    train_scores = np.column_stack(train_scores)
    test_scores = np.column_stack(test_scores)

    # Standardize the scores of different models before combining.
    train_scores_norm, test_scores_norm = standardizer(
        train_scores,
        test_scores,
    )

    combinations = [
        (
            "Average",
            average(train_scores_norm),
            average(test_scores_norm),
        ),
        (
            "Maximization",
            maximization(train_scores_norm),
            maximization(test_scores_norm),
        ),
    ]

    print("\nCombined scores:")
    print(f"{'Strategy':<16} {'Train BA':>12} {'Test BA':>12}")

    for name, train_combined, test_combined in combinations:
        train_combined = np.asarray(train_combined).ravel()
        test_combined = np.asarray(test_combined).ravel()

        # Choose the threshold using only the training scores.
        threshold = np.quantile(
            train_combined,
            1 - contamination,
        )

        train_pred = (train_combined >= threshold).astype(int)
        test_pred = (test_combined >= threshold).astype(int)

        train_ba = balanced_accuracy_score(y_train, train_pred)
        test_ba = balanced_accuracy_score(y_test, test_pred)

        print(f"{name:<16} {train_ba:>12.4f} {test_ba:>12.4f}")


def exercise4():
    print("\nEXERCISE 4: CARDIO DATASET")

    file_path = find_cardio_file()

    if file_path is None:
        print("Cardio dataset not found; skipping Exercise 4.")
        print("Download cardio.mat from:")
        print("https://odds.cs.stonybrook.edu/cardiotocogrpahy-dataset/")
        print("Place cardio.mat in the same folder as lab2.py.")
        return

    X, y = load_cardio_dataset(file_path)
    contamination = float(np.mean(y == 1))

    print(f"Samples: {len(y)}")
    print(f"Features: {X.shape[1]}")
    print(f"Contamination: {contamination:.4f}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y,
    )

    # Normalize using training data statistics.
    X_train, X_test = standardizer(X_train, X_test)

    evaluate_ensemble(
        "KNN",
        X_train,
        X_test,
        y_train,
        y_test,
        contamination,
    )

    evaluate_ensemble(
        "LOF",
        X_train,
        X_test,
        y_train,
        y_test,
        contamination,
    )

if __name__ == "__main__":
    exercise1()
    exercise2()
    exercise3()
    exercise4()

    # Display all figures after the exercises have run.
    plt.show()