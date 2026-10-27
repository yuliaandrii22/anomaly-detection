    #Anomaly Detection - Lab 1

import numpy as np
import matplotlib.pyplot as plt

from pyod.utils.data import generate_data
from pyod.models.knn import KNN

from sklearn.metrics import (
    confusion_matrix,
    balanced_accuracy_score,
    roc_curve,
    roc_auc_score,
)


def exercise_1():
    #Genereaza si reprezinta date 2D pentru antrenare si testare

    print("EXERCITIUL 1 - Generarea datelor")

    X_train, X_test, y_train, y_test = generate_data(
        n_train=400,
        n_test=100,
        n_features=2,
        contamination=0.1,
        random_state=42,
    )

    normal = X_train[y_train == 0]
    outliers = X_train[y_train == 1]

    plt.figure(figsize=(8, 6))
    plt.scatter(
        normal[:, 0], normal[:, 1],
        color="royalblue", label="Date normale", alpha=0.7
    )
    plt.scatter(
        outliers[:, 0], outliers[:, 1],
        color="crimson", label="Anomalii", alpha=0.8
    )
    plt.title("Exercitiul 1 - Datele de antrenare")
    plt.xlabel("Caracteristica 1")
    plt.ylabel("Caracteristica 2")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    print("Dimensiunea setului de antrenare:", X_train.shape)
    print("Dimensiunea setului de testare:", X_test.shape)
    print("Anomalii in setul de antrenare:", int(np.sum(y_train == 1)))
    print("Anomalii in setul de testare:", int(np.sum(y_test == 1)))

    return X_train, X_test, y_train, y_test


def exercise_2(X_train, X_test, y_train, y_test):
    #Detecteaza anomalii cu KNN si evalueaza rezultatele
    print("EXERCITIUL 2 - Detectarea anomaliilor cu KNN")

    model = KNN(contamination=0.1)
    model.fit(X_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_test_scores = model.decision_function(X_test)

    cm = confusion_matrix(y_test, y_test_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    tpr = tp / (tp + fn) if tp + fn else 0.0
    tnr = tn / (tn + fp) if tn + fp else 0.0
    fpr_value = fp / (fp + tn) if fp + tn else 0.0
    ba = balanced_accuracy_score(y_test, y_test_pred)

    fpr_curve, tpr_curve, _ = roc_curve(y_test, y_test_scores)
    auc = roc_auc_score(y_test, y_test_scores)

    print("\nMatricea de confuzie:")
    print(cm)
    print(f"TN = {tn}, FP = {fp}, FN = {fn}, TP = {tp}")
    print(f"TPR / Recall: {tpr:.4f}")
    print(f"TNR / Specificity: {tnr:.4f}")
    print(f"FPR: {fpr_value:.4f}")
    print(f"Balanced Accuracy: {ba:.4f}")
    print(f"AUC: {auc:.4f}")

    plt.figure(figsize=(8, 6))
    plt.plot(fpr_curve, tpr_curve, label=f"KNN (AUC = {auc:.3f})")
    plt.plot(
        [0, 1], [0, 1],
        linestyle="--", label="Clasificator aleator"
    )
    plt.title("Exercitiul 2 - Curba ROC")
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    train_ba = balanced_accuracy_score(y_train, y_train_pred)
    print(f"\nBalanced Accuracy pe antrenare: {train_ba:.4f}")

    print("\nComparatie pentru rate diferite de contaminare:")

    for rate in (0.05, 0.10, 0.20):
        model_temp = KNN(contamination=rate)
        model_temp.fit(X_train)
        predictions = model_temp.predict(X_test)
        score = balanced_accuracy_score(y_test, predictions)

        print(
            f"Contamination = {rate:.2f}"
            f" -> Balanced Accuracy = {score:.4f}"
        )


def exercise_3():
    #Detecteaza anomalii unidimensionale folosind Z-score

    print("EXERCITIUL 3 - Z-score unidimensional")

    contamination = 0.1

    X_train, _, y_train, _ = generate_data(
        n_train=1000,
        n_test=0,
        n_features=1,
        contamination=contamination,
        random_state=42,
    )

    x = X_train.ravel()
    mean = np.mean(x)
    std = np.std(x)

    if std == 0:
        raise ValueError("Abaterea standard este zero.")

    z_scores = np.abs((x - mean) / std)
    threshold = np.quantile(z_scores, 1 - contamination)
    y_pred = (z_scores >= threshold).astype(int)

    ba = balanced_accuracy_score(y_train, y_pred)
    cm = confusion_matrix(y_train, y_pred, labels=[0, 1])

    print(f"Numar total de exemple: {len(x)}")
    print(f"Media: {mean:.4f}")
    print(f"Abaterea standard: {std:.4f}")
    print(f"Pragul Z-score: {threshold:.4f}")
    print("Anomalii reale:", int(np.sum(y_train == 1)))
    print("Anomalii detectate:", int(np.sum(y_pred == 1)))
    print(f"Balanced Accuracy: {ba:.4f}")
    print("\nMatricea de confuzie:")
    print(cm)

    plt.figure(figsize=(10, 5))
    plt.scatter(
        np.arange(len(x)), x,
        c=y_pred, cmap="coolwarm", alpha=0.75
    )
    plt.title("Exercitiul 3 - Detectie cu Z-score")
    plt.xlabel("Indexul exemplului")
    plt.ylabel("Valoare")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.figure(figsize=(8, 5))
    plt.hist(z_scores, bins=40)
    plt.axvline(
        threshold, linestyle="--",
        label=f"Prag = {threshold:.2f}"
    )
    plt.title("Distributia valorilor absolute Z-score")
    plt.xlabel("|Z-score|")
    plt.ylabel("Frecventa")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


def exercise_4():
    #Genereaza date multidimensionale si aplica scorul Mahalanobis
    print("EXERCITIUL 4 - Z-score multidimensional")

    rng = np.random.default_rng(42)

    n_samples = 1000
    contamination = 0.1
    n_outliers = int(n_samples * contamination)
    n_normal = n_samples - n_outliers

    mu = np.array([2.0, 3.0])
    covariance = np.array([
        [2.0, 0.8],
        [0.8, 1.0],
    ])

    # Descompunerea Cholesky: covariance = L @ L.T.
    L = np.linalg.cholesky(covariance)

    X_standard = rng.standard_normal(size=(n_normal, 2))
    X_normal = X_standard @ L.T + mu

    X_outliers = rng.uniform(
        low=[-5.0, -5.0],
        high=[10.0, 10.0],
        size=(n_outliers, 2),
    )

    X = np.vstack((X_normal, X_outliers))

    y_true = np.concatenate((
        np.zeros(n_normal, dtype=int),
        np.ones(n_outliers, dtype=int),
    ))

    order = rng.permutation(n_samples)
    X = X[order]
    y_true = y_true[order]

    # Scorul Mahalanobis la patrat.
    centered = X - mu
    solved = np.linalg.solve(covariance, centered.T).T
    z_scores = np.sum(centered * solved, axis=1)

    threshold = np.quantile(z_scores, 1 - contamination)
    y_pred = (z_scores >= threshold).astype(int)

    ba = balanced_accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    print("Vectorul mediei:")
    print(mu)
    print("\nMatricea de covarianta:")
    print(covariance)
    print("\nMatricea Cholesky L:")
    print(L)
    print("\nVerificare Sigma = L @ L.T:",
          np.allclose(covariance, L @ L.T))
    print("\nNumar total de exemple:", len(X))
    print("Numar de exemple normale:", n_normal)
    print("Anomalii reale:", int(np.sum(y_true == 1)))
    print("Anomalii detectate:", int(np.sum(y_pred == 1)))
    print(f"Prag Z-score multidimensional: {threshold:.4f}")
    print(f"Balanced Accuracy: {ba:.4f}")
    print("\nMatricea de confuzie:")
    print(cm)

    plt.figure(figsize=(8, 6))
    plt.scatter(
        X[y_true == 0, 0], X[y_true == 0, 1],
        color="royalblue", label="Date normale", alpha=0.6
    )
    plt.scatter(
        X[y_true == 1, 0], X[y_true == 1, 1],
        color="crimson", label="Anomalii reale", alpha=0.7
    )
    plt.title("Exercitiul 4 - Date generate")
    plt.xlabel("Caracteristica 1")
    plt.ylabel("Caracteristica 2")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(
        X[:, 0], X[:, 1],
        c=y_true, cmap="coolwarm", alpha=0.7
    )
    axes[0].set_title("Etichete reale")
    axes[0].set_xlabel("Caracteristica 1")
    axes[0].set_ylabel("Caracteristica 2")

    axes[1].scatter(
        X[:, 0], X[:, 1],
        c=y_pred, cmap="coolwarm", alpha=0.7
    )
    axes[1].set_title("Etichete prezise cu Z-score")
    axes[1].set_xlabel("Caracteristica 1")
    axes[1].set_ylabel("Caracteristica 2")

    for ax in axes:
        ax.grid(True, alpha=0.3)

    fig.tight_layout()


def main():
    X_train, X_test, y_train, y_test = exercise_1()
    exercise_2(X_train, X_test, y_train, y_test)
    exercise_3()
    exercise_4()

    # Afiseaza toate figurile impreuna, fara pauze intre grafice.
    plt.show()

if __name__ == "__main__":
    main()
