"""
Mall Customer Segmentation Using K-Means Clustering

This script performs exploratory data analysis and customer segmentation
using the Mall Customer Segmentation dataset.

Dataset:
https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python

Before running:
1. Download Mall_Customers.csv from Kaggle.
2. Place it inside the data/ folder.
3. Run this script from the project root directory.

Example:
python src/customer_segmentation.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans


# -------------------------------------------------------------------
# Project paths
# -------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "Mall_Customers.csv"
IMAGES_DIR = PROJECT_ROOT / "images"

IMAGES_DIR.mkdir(exist_ok=True)


# -------------------------------------------------------------------
# Load and prepare data
# -------------------------------------------------------------------

def load_data(path: Path) -> pd.DataFrame:
    """Load the Mall Customers dataset."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. "
            "Please download Mall_Customers.csv and place it in the data/ folder."
        )

    df = pd.read_csv(path)

    df = df.rename(
        columns={
            "Spending Score (1-100)": "SpendingScore",
            "Annual Income (k$)": "AnnualIncome",
        }
    )

    return df


# -------------------------------------------------------------------
# Exploratory data analysis
# -------------------------------------------------------------------

def explore_data(df: pd.DataFrame) -> None:
    """Print basic information about the dataset."""
    print("\nFirst 5 rows:")
    print(df.head())

    print("\nDataset shape:")
    print(df.shape)

    print("\nSummary statistics:")
    print(df.describe())

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nData types:")
    print(df.dtypes)


def create_pairplot(df: pd.DataFrame) -> None:
    """Create and save a pairplot of numerical features."""
    sns.pairplot(df[["Age", "AnnualIncome", "SpendingScore"]])
    plt.savefig(IMAGES_DIR / "pairplot.png", bbox_inches="tight")
    plt.close()


def plot_income_vs_spending(df: pd.DataFrame) -> None:
    """Create and save a scatter plot of annual income vs spending score."""
    plt.figure(figsize=(10, 6))
    plt.scatter(df["AnnualIncome"], df["SpendingScore"], s=50)
    plt.xlabel("Annual Income (k$)")
    plt.ylabel("Spending Score (1-100)")
    plt.title("Annual Income vs Spending Score")
    plt.savefig(IMAGES_DIR / "income_spending_scatter.png", bbox_inches="tight")
    plt.close()


# -------------------------------------------------------------------
# K-Means clustering
# -------------------------------------------------------------------

def calculate_wcss(features: pd.DataFrame, max_clusters: int = 10) -> list[float]:
    """Calculate WCSS values for the elbow method."""
    wcss = []

    for i in range(1, max_clusters + 1):
        kmeans = KMeans(
            n_clusters=i,
            init="k-means++",
            max_iter=300,
            n_init=10,
            random_state=0,
        )
        kmeans.fit(features)
        wcss.append(kmeans.inertia_)

    return wcss


def plot_elbow_method(wcss: list[float], filename: str) -> None:
    """Create and save an elbow method plot."""
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(wcss) + 1), wcss, marker="o")
    plt.xlabel("Number of Clusters")
    plt.ylabel("WCSS")
    plt.title("Elbow Method to Determine Optimal Number of Clusters")
    plt.savefig(IMAGES_DIR / filename, bbox_inches="tight")
    plt.close()


def apply_kmeans(features: pd.DataFrame, n_clusters: int) -> tuple[KMeans, list[int]]:
    """Fit a K-Means model and return the model and cluster labels."""
    kmeans = KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        max_iter=300,
        n_init=10,
        random_state=0,
    )

    labels = kmeans.fit_predict(features)

    return kmeans, labels


def plot_income_spending_clusters(
    features: pd.DataFrame,
    labels: list[int],
    model: KMeans,
) -> None:
    """Create and save a scatter plot of income/spending clusters."""
    plt.figure(figsize=(10, 6))
    plt.scatter(
        features.iloc[:, 0],
        features.iloc[:, 1],
        c=labels,
        s=50,
        cmap="viridis",
    )

    centers = model.cluster_centers_
    plt.scatter(
        centers[:, 0],
        centers[:, 1],
        c="red",
        s=200,
        alpha=0.75,
        marker="X",
        label="Centroids",
    )

    plt.xlabel("Annual Income (k$)")
    plt.ylabel("Spending Score (1-100)")
    plt.title("Customer Segments Based on Income and Spending Score")
    plt.legend()
    plt.savefig(IMAGES_DIR / "income_spending_clusters.png", bbox_inches="tight")
    plt.close()


def plot_age_spending_clusters(
    features: pd.DataFrame,
    labels: list[int],
    model: KMeans,
) -> None:
    """Create and save a scatter plot of age/spending clusters."""
    plt.figure(figsize=(10, 6))
    plt.scatter(
        features.iloc[:, 0],
        features.iloc[:, 1],
        c=labels,
        s=50,
        cmap="viridis",
    )

    centers = model.cluster_centers_
    plt.scatter(
        centers[:, 0],
        centers[:, 1],
        c="red",
        s=200,
        alpha=0.75,
        marker="X",
        label="Centroids",
    )

    plt.xlabel("Age")
    plt.ylabel("Spending Score (1-100)")
    plt.title("Customer Segments Based on Age and Spending Score")
    plt.legend()
    plt.savefig(IMAGES_DIR / "age_spending_clusters.png", bbox_inches="tight")
    plt.close()


def plot_3d_clusters(df: pd.DataFrame) -> None:
    """Create and save a 3D cluster plot using age, income, and spending score."""
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        df["Age"],
        df["AnnualIncome"],
        df["SpendingScore"],
        c=df["ClusterAgeSpendIncome"],
        s=50,
        cmap="viridis",
    )

    ax.set_xlabel("Age")
    ax.set_ylabel("Annual Income (k$)")
    ax.set_zlabel("Spending Score (1-100)")
    plt.title("Customer Segments Based on Age, Income, and Spending Score")
    plt.savefig(IMAGES_DIR / "age_income_spending_3d_clusters.png", bbox_inches="tight")
    plt.close()


# -------------------------------------------------------------------
# Main workflow
# -------------------------------------------------------------------

def main() -> None:
    """Run the full customer segmentation workflow."""
    df = load_data(DATA_PATH)

    explore_data(df)

    create_pairplot(df)
    plot_income_vs_spending(df)

    # Primary clustering: Annual Income and Spending Score
    income_spending_features = df[["AnnualIncome", "SpendingScore"]]

    income_spending_wcss = calculate_wcss(income_spending_features)
    plot_elbow_method(income_spending_wcss, "elbow_method_income_spending.png")

    income_spending_model, income_spending_labels = apply_kmeans(
        income_spending_features,
        n_clusters=5,
    )

    df["ClusterIncomeSpending"] = income_spending_labels

    plot_income_spending_clusters(
        income_spending_features,
        income_spending_labels,
        income_spending_model,
    )

    # Additional clustering: Age and Spending Score
    age_spending_features = df[["Age", "SpendingScore"]]

    age_spending_wcss = calculate_wcss(age_spending_features)
    plot_elbow_method(age_spending_wcss, "elbow_method_age_spending.png")

    age_spending_model, age_spending_labels = apply_kmeans(
        age_spending_features,
        n_clusters=4,
    )

    df["ClusterAgeSpending"] = age_spending_labels

    plot_age_spending_clusters(
        age_spending_features,
        age_spending_labels,
        age_spending_model,
    )

    # Additional clustering: Age, Annual Income, and Spending Score
    age_income_spending_features = df[["Age", "AnnualIncome", "SpendingScore"]]

    age_income_spending_wcss = calculate_wcss(age_income_spending_features)
    plot_elbow_method(
        age_income_spending_wcss,
        "elbow_method_age_income_spending.png",
    )

    age_income_spending_model, age_income_spending_labels = apply_kmeans(
        age_income_spending_features,
        n_clusters=6,
    )

    df["ClusterAgeSpendIncome"] = age_income_spending_labels

    plot_3d_clusters(df)

    output_path = PROJECT_ROOT / "data" / "mall_customers_with_clusters.csv"
    df.to_csv(output_path, index=False)

    print("\nProject complete.")
    print(f"Clustered dataset saved to: {output_path}")
    print(f"Visualisations saved to: {IMAGES_DIR}")


if __name__ == "__main__":
    main()
