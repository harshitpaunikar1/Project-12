"""
PCA case study on housing data.
Reduces high-dimensional housing features to principal components for visualization and regression.
"""
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.metrics import r2_score, mean_squared_error
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class HousingPCAAnalysis:
    """
    Performs PCA on housing features and compares regression
    on raw features vs principal components.
    """

    def __init__(self, feature_cols: List[str], target_col: str = "price",
                 n_components: Optional[int] = None, variance_threshold: float = 0.95):
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.n_components = n_components
        self.variance_threshold = variance_threshold
        self.scaler = None
        self.pca = None
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> "HousingPCAAnalysis":
        """Fit scaler and PCA on the feature columns."""
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn required.")
        X = df[self.feature_cols].fillna(df[self.feature_cols].median())
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        n = self.n_components or X_scaled.shape[1]
        self.pca = PCA(n_components=n)
        self.pca.fit(X_scaled)
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Return PCA-transformed features for the given DataFrame."""
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")
        X = df[self.feature_cols].fillna(df[self.feature_cols].median())
        X_scaled = self.scaler.transform(X)
        return self.pca.transform(X_scaled)

    def select_components_by_variance(self) -> int:
        """Return the number of components explaining >= variance_threshold of total variance."""
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        cumsum = np.cumsum(self.pca.explained_variance_ratio_)
        n = int(np.searchsorted(cumsum, self.variance_threshold) + 1)
        return min(n, len(cumsum))

    def explained_variance_table(self) -> pd.DataFrame:
        """Return a DataFrame with per-component variance explained."""
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        evr = self.pca.explained_variance_ratio_
        return pd.DataFrame({
            "component": [f"PC{i+1}" for i in range(len(evr))],
            "explained_variance_pct": (evr * 100).round(2),
            "cumulative_pct": (np.cumsum(evr) * 100).round(2),
        })

    def loadings_table(self) -> pd.DataFrame:
        """Return feature loadings for each principal component."""
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        n_components = self.pca.components_.shape[0]
        return pd.DataFrame(
            self.pca.components_.T,
            index=self.feature_cols,
            columns=[f"PC{i+1}" for i in range(n_components)],
        ).round(4)

    def top_loading_features(self, component_idx: int = 0, top_n: int = 5) -> pd.Series:
        """Return features with highest absolute loadings for a given component."""
        loadings = self.loadings_table()
        col = f"PC{component_idx + 1}"
        return loadings[col].abs().sort_values(ascending=False).head(top_n)

    def compare_regression(self, df: pd.DataFrame, test_size: float = 0.2) -> pd.DataFrame:
        """
        Compare Ridge regression performance on raw features vs PCA components.
        Returns a comparison DataFrame with R2 and RMSE.
        """
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn required.")
        if not self._fitted:
            self.fit(df)
        X_raw = df[self.feature_cols].fillna(df[self.feature_cols].median())
        y = df[self.target_col]
        X_pca = self.transform(df)
        n_pca = self.select_components_by_variance()
        X_pca_reduced = X_pca[:, :n_pca]

        results = []
        for name, X in [("Raw Features", X_raw), ("PCA Components", X_pca_reduced)]:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            if name == "Raw Features":
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)
            model = Ridge(alpha=1.0)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            r2 = float(r2_score(y_test, preds))
            rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
            results.append({
                "method": name,
                "n_features": X_train.shape[1],
                "r2": round(r2, 4),
                "rmse": round(rmse, 2),
            })
        return pd.DataFrame(results)

    def biplot_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
        """Return scores (Nx2) and loadings DataFrame for PC1 vs PC2 biplot."""
        scores = self.transform(df)[:, :2]
        loadings = self.loadings_table()[["PC1", "PC2"]]
        return scores, loadings


if __name__ == "__main__":
    np.random.seed(42)
    n = 1000
    sqft = np.random.uniform(500, 5000, n)
    bedrooms = np.random.randint(1, 6, n)
    bathrooms = np.random.randint(1, 4, n)
    age = np.random.randint(0, 60, n)
    garage = np.random.randint(0, 3, n)
    lot_size = np.random.uniform(1000, 20000, n)
    school_rating = np.random.uniform(1, 10, n)
    crime_rate = np.random.uniform(0, 10, n)
    dist_cbd = np.random.uniform(1, 50, n)
    price = (150 * sqft + 20000 * bedrooms + 15000 * bathrooms
             - 1000 * age + 5000 * garage + 2000 * school_rating
             - 3000 * crime_rate - 1000 * dist_cbd
             + np.random.normal(0, 30000, n))

    df = pd.DataFrame({
        "sqft": sqft, "bedrooms": bedrooms.astype(float), "bathrooms": bathrooms.astype(float),
        "age": age.astype(float), "garage": garage.astype(float), "lot_size": lot_size,
        "school_rating": school_rating, "crime_rate": crime_rate, "dist_cbd": dist_cbd,
        "price": price,
    })

    features = ["sqft", "bedrooms", "bathrooms", "age", "garage",
                "lot_size", "school_rating", "crime_rate", "dist_cbd"]
    analysis = HousingPCAAnalysis(feature_cols=features, target_col="price")
    analysis.fit(df)

    ev_table = analysis.explained_variance_table()
    print("Explained variance per component:")
    print(ev_table.to_string(index=False))
    print(f"\nComponents for {analysis.variance_threshold*100:.0f}% variance: "
          f"{analysis.select_components_by_variance()}")

    print("\nTop loadings for PC1:")
    print(analysis.top_loading_features(component_idx=0, top_n=5))

    comparison = analysis.compare_regression(df)
    print("\nRegression comparison:")
    print(comparison.to_string(index=False))
