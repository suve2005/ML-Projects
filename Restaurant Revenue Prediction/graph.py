"""
Visualization module for plotting features against restaurant revenue.

This module provides functions to visualize the relationship between
all numeric features and the target variable (revenue) in the restaurant
dataset.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path


def load_and_preprocess_data(filepath=None, reference_date="2026-01-01"):
    """
    Load and preprocess the restaurant data.

    Parameters:
    -----------
    filepath : str or Path, optional
        Path to the CSV file. Defaults to data/train.csv relative to script.
    reference_date : str, optional
        Reference date for age calculation in YYYY-MM-DD format.
        Default is "2026-01-01".

    Returns:
    --------
    pd.DataFrame
        Preprocessed dataframe with engineered features.
    """
    if filepath is None:
        # Try to get script directory, fall back to current directory
        if "__file__" in globals():
            script_dir = Path(__file__).parent
        else:
            script_dir = Path.cwd()
        filepath = script_dir / "data" / "train.csv"

    df = pd.read_csv(filepath)

    # Calculate restaurant age
    def calculate_restaurant_age(open_date, ref_date=reference_date):
        open_date_parsed = pd.to_datetime(open_date, format="%m/%d/%Y")
        reference = pd.to_datetime(ref_date)
        age_years = (reference - open_date_parsed).days / 365.25
        return max(0, age_years)

    df["age"] = df["Open Date"].apply(calculate_restaurant_age)

    # Create one-hot encoded columns
    df["CityGroup_Big_Cities"] = (df["City Group"] == "Big Cities").astype(int)
    df["Type_FC"] = (df["Type"] == "FC").astype(int)
    df["Type_IL"] = (df["Type"] == "IL").astype(int)

    return df


def plot_features_vs_revenue(df=None, filepath=None, reference_date="2026-01-01",
                              max_cols_per_row=5, figsize=None):
    """
    Create a grid of scatter plots showing all features against revenue.

    Parameters:
    -----------
    df : pd.DataFrame, optional
        Preprocessed dataframe. If None, will be loaded from filepath.
    filepath : str or Path, optional
        Path to the CSV file (used if df is None).
    reference_date : str, optional
        Reference date for age calculation. Default "2026-01-01".
    max_cols_per_row : int, optional
        Maximum number of columns per row in the subplot grid. Default 5.
    figsize : tuple, optional
        Figure size as (width, height). If None, will be calculated based
        on number of features.

    Returns:
    --------
    pd.DataFrame
        The dataframe used for plotting.
    """
    # Load data if not provided
    if df is None:
        df = load_and_preprocess_data(filepath, reference_date)

    # Define feature columns to plot
    # Numeric P-features (P1-P37)
    p_features = [f"P{i}" for i in range(1, 38)]

    # Engineered features (excluding revenue)
    other_features = ["age", "CityGroup_Big_Cities", "Type_FC", "Type_IL"]

    # All features to plot
    features_to_plot = p_features + other_features

    # Filter to only include columns that actually exist
    available_features = [f for f in features_to_plot if f in df.columns]

    # Calculate number of subplots needed
    n_features = len(available_features)
    n_cols = min(max_cols_per_row, n_features)
    n_rows = int(np.ceil(n_features / n_cols))

    # Set figure size
    if figsize is None:
        figsize = (n_cols * 3.5, n_rows * 3.0)

    # Create the subplot grid
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.atleast_1d(axes).flatten()

    # Set up seaborn style
    sns.set_style("whitegrid")

    # Plot each feature vs revenue
    for idx, feature in enumerate(available_features):
        ax = axes[idx]

        # Scatter plot
        ax.scatter(df[feature], df["revenue"], alpha=0.6, c="steelblue",
                   edgecolor="w", s=40)

        # Add trend line if there's enough variation
        try:
            x = df[feature].values
            y = df["revenue"].values
            if len(x) > 1 and np.std(x) > 0:
                mask = ~(np.isnan(x) | np.isnan(y))
                if np.sum(mask) > 1:
                    z = np.polyfit(x[mask], y[mask], 1)
                    p = np.poly1d(z)
                    x_line = np.linspace(np.min(x), np.max(x), 100)
                    ax.plot(x_line, p(x_line), color="red", linewidth=2,
                           label="Trend")
        except Exception:
            pass

        ax.set_xlabel(feature, fontsize=9)
        ax.set_ylabel("Revenue (USD)", fontsize=9)
        ax.set_title(f"{feature} vs Revenue", fontsize=10, fontweight="bold")
        ax.tick_params(axis='both', labelsize=8)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n_features, len(axes)):
        axes[idx].set_visible(False)

    # Add overall title
    fig.suptitle("All Features vs Revenue", fontsize=14, fontweight="bold", y=1.02)

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)

    return df, fig


def plot_correlation_heatmap(df=None, filepath=None, reference_date="2026-01-01",
                              top_n=20):
    """
    Create a correlation heatmap for features vs revenue.

    Parameters:
    -----------
    df : pd.DataFrame, optional
        Preprocessed dataframe.
    filepath : str or Path, optional
        Path to the CSV file (used if df is None).
    reference_date : str, optional
        Reference date for age calculation.
    top_n : int, optional
        Number of top correlated features to include. Default 20.

    Returns:
    --------
    pd.DataFrame
        The correlation matrix subset.
    """
    # Load data if not provided
    if df is None:
        df = load_and_preprocess_data(filepath, reference_date)

    # Define feature columns
    p_features = [f"P{i}" for i in range(1, 38)]
    other_features = ["age", "CityGroup_Big_Cities", "Type_FC", "Type_IL"]
    features_to_plot = p_features + other_features
    available_features = [f for f in features_to_plot if f in df.columns]

    # Calculate correlation matrix
    corr_features = available_features + ["revenue"]
    corr_matrix = df[corr_features].corr()

    # Sort by correlation with revenue and take top_n
    revenue_corr = corr_matrix["revenue"].drop("revenue").abs().sort_values(ascending=False)
    top_features = revenue_corr.head(top_n).index.tolist()

    # Create heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix.loc[top_features + ["revenue"], top_features + ["revenue"]],
                annot=True, cmap="coolwarm", center=0, fmt=".2f",
                linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title(f"Correlation Heatmap (Top {top_n} Features vs Revenue)",
              fontsize=14, fontweight="bold")
    plt.tight_layout()

    return corr_matrix


def plot_categorical_vs_revenue(df=None, filepath=None, reference_date="2026-01-01"):
    """
    Create box plots for categorical features vs revenue.

    Parameters:
    -----------
    df : pd.DataFrame, optional
        Preprocessed dataframe.
    filepath : str or Path, optional
        Path to the CSV file (used if df is None).
    reference_date : str, optional
        Reference date for age calculation.

    Returns:
    --------
    pd.DataFrame
        The dataframe used for plotting.
    """
    if df is None:
        df = load_and_preprocess_data(filepath, reference_date)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # City Group vs Revenue
    city_group_order = df.groupby("City Group")["revenue"].mean().sort_values(ascending=False).index
    sns.boxplot(data=df, x="City Group", y="revenue", ax=axes[0], order=city_group_order)
    axes[0].set_title("City Group vs Revenue", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("City Group", fontsize=10)
    axes[0].set_ylabel("Revenue (USD)", fontsize=10)
    axes[0].tick_params(axis='x', rotation=15)

    # Type vs Revenue
    type_order = df.groupby("Type")["revenue"].mean().sort_values(ascending=False).index
    sns.boxplot(data=df, x="Type", y="revenue", ax=axes[1], order=type_order)
    axes[1].set_title("Restaurant Type vs Revenue", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Type", fontsize=10)
    axes[1].set_ylabel("Revenue (USD)", fontsize=10)

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)

    return df, fig


def plot_cost_comparison(cost_history_train, cost_history_cv, title="Cost Comparison: J_train vs J_cv"):
    """
    Plot the cost function over iterations for training and CV data.

    Parameters:
    -----------
    cost_history_train : list or np.ndarray
        Cost values from training set at each iteration.
    cost_history_cv : list or np.ndarray
        Cost values from validation/cross-validation set at each iteration.
    title : str, optional
        Title for the plot. Default is "Cost Comparison: J_train vs J_cv".

    Returns:
    --------
    matplotlib.figure.Figure
        The figure object containing the plot.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    iterations = range(1, len(cost_history_train) + 1)

    ax.plot(iterations, cost_history_train, linewidth=2, color="steelblue",
            label="J_train", marker='o', markersize=3, markevery=50)
    ax.plot(iterations, cost_history_cv, linewidth=2, color="darkorange",
            label="J_cv", marker='s', markersize=3, markevery=50)

    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Cost J(W,b)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def train_and_evaluate_on_splits(train_df, cv_df, learning_rate=0.01, n_iterations=1000):
    """
    Train linear regression model and evaluate on both training and CV sets.

    This function runs gradient descent on the training data and tracks the cost
    on both the training and CV sets at each iteration, enabling a comparison
    of J_train vs J_cv.

    Parameters:
    -----------
    train_df : pd.DataFrame
        Training data with features and 'revenue' target column.
    cv_df : pd.DataFrame
        Cross-validation data with features and 'revenue' target column.
    learning_rate : float, optional
        Learning rate for gradient descent. Default is 0.01.
    n_iterations : int, optional
        Number of iterations for gradient descent. Default is 1000.

    Returns:
    --------
    tuple
        (W, b, cost_history_train, cost_history_cv, y_train_pred, y_cv_pred)
        where:
        - W: learned weights
        - b: learned bias
        - cost_history_train: list of cost values on training set per iteration
        - cost_history_cv: list of cost values on CV set per iteration
        - y_train_pred: predictions on training set
        - y_cv_pred: predictions on CV set
    """
    # Get feature columns (all except 'revenue')
    feature_cols = [col for col in train_df.columns if col != 'revenue']

    # Extract feature matrices and targets
    X_train = train_df[feature_cols].values.copy()
    y_train = train_df['revenue'].values.copy()
    X_cv = cv_df[feature_cols].values.copy()
    y_cv = cv_df['revenue'].values.copy()

    n_samples, n_features = X_train.shape

    # Initialize weights W as row vector (1 x n_features) and bias b as scalar
    W = np.zeros((1, n_features))
    b = 0.0

    # Reshape y values to column vectors for computation
    y_train_col = y_train.reshape(-1, 1)
    y_cv_col = y_cv.reshape(-1, 1)

    cost_history_train = []
    cost_history_cv = []

    print(f"\n--- Training Linear Regression with Train/CV Cost Tracking ---")
    print(f"Training samples: {n_samples}")
    print(f"CV samples: {len(y_cv)}")
    print(f"Features: {n_features}")
    print(f"Learning rate: {learning_rate}")
    print(f"Iterations: {n_iterations}")

    for i in range(n_iterations):
        # Forward pass on training data
        y_train_pred_col = X_train @ W.T + b  # Shape: (n_samples, 1)

        # Compute training cost: J_train = 1/(2*m) * sum((y_train - y_train_pred)^2)
        m_train = n_samples
        cost_train = (1 / (2 * m_train)) * np.sum((y_train_col - y_train_pred_col) ** 2)
        cost_history_train.append(cost_train)

        # Forward pass on CV data
        y_cv_pred_col = X_cv @ W.T + b  # Shape: (n_cv_samples, 1)

        # Compute CV cost: J_cv = 1/(2*m) * sum((y_cv - y_cv_pred)^2)
        m_cv = len(y_cv)
        cost_cv = (1 / (2 * m_cv)) * np.sum((y_cv_col - y_cv_pred_col) ** 2)
        cost_history_cv.append(cost_cv)

        # Gradient descent using training data gradients
        error_train = y_train_pred_col - y_train_col
        dW = (1 / m_train) * (X_train.T @ error_train)
        db = (1 / m_train) * np.sum(error_train)

        # Update weights and bias
        W = W - learning_rate * dW.T
        b = b - learning_rate * db

        # Print progress every 100 iterations
        if (i + 1) % 100 == 0 or i == 0:
            print(f"Iteration {i+1}: J_train = {cost_train:,.2f}, J_cv = {cost_cv:,.2f}")

    # Final predictions
    y_train_pred = (X_train @ W.T + b).flatten()
    y_cv_pred = (X_cv @ W.T + b).flatten()

    return W, b, cost_history_train, cost_history_cv, y_train_pred, y_cv_pred, feature_cols


def plot_j_train_vs_j_cv(train_df=None, cv_df=None, filepath=None, reference_date="2026-01-01",
                          learning_rate=0.01, n_iterations=1000):
    """
    Load train and CV split data, train a linear regression model,
    and plot J_train vs J_cv over iterations.

    Parameters:
    -----------
    train_df : pd.DataFrame, optional
        Preprocessed training dataframe. If None, will be loaded from filepath.
    cv_df : pd.DataFrame, optional
        Preprocessed CV dataframe. If None, will be loaded from filepath.
    filepath : str or Path, optional
        Path to the directory containing train_split.csv and cv_split.csv.
        If None, uses data/ directory relative to script.
    reference_date : str, optional
        Reference date for age calculation. Default "2026-01-01".
    learning_rate : float, optional
        Learning rate for gradient descent. Default 0.01.
    n_iterations : int, optional
        Number of iterations for gradient descent. Default 1000.

    Returns:
    --------
    tuple
        (W, b, cost_history_train, cost_history_cv, y_train_pred, y_cv_pred, feature_cols)
    """
    # Determine paths
    if filepath is None:
        if "__file__" in globals():
            script_dir = Path(__file__).parent
        else:
            script_dir = Path.cwd()
        data_dir = script_dir / "data"
    else:
        data_dir = Path(filepath)

    # Load data if not provided
    if train_df is None:
        train_path = data_dir / "train_split.csv"
        train_df = pd.read_csv(train_path)
        print(f"Loaded training data from: {train_path}")

    if cv_df is None:
        cv_path = data_dir / "cv_split.csv"
        cv_df = pd.read_csv(cv_path)
        print(f"Loaded CV data from: {cv_path}")

    # Train and evaluate
    result = train_and_evaluate_on_splits(train_df, cv_df, learning_rate, n_iterations)
    W, b, cost_history_train, cost_history_cv, y_train_pred, y_cv_pred, feature_cols = result

    # Plot J_train vs J_cv
    fig = plot_cost_comparison(cost_history_train, cost_history_cv)

    # Print summary statistics
    print(f"\n=== Final Summary ===")
    print(f"Final J_train: {cost_history_train[-1]:,.2f}")
    print(f"Final J_cv: {cost_history_cv[-1]:,.2f}")
    print(f"Final bias b: {b:,.2f}")

    # Compute RMSE for both sets
    rmse_train = np.sqrt(cost_history_train[-1] * 2)  # RMSE = sqrt(2 * cost)
    rmse_cv = np.sqrt(cost_history_cv[-1] * 2)
    print(f"RMSE (Training): {rmse_train:,.2f}")
    print(f"RMSE (CV): {rmse_cv:,.2f}")

    # Compute R²
    y_train_true = train_df['revenue'].values
    y_cv_true = cv_df['revenue'].values

    ss_res_train = np.sum((y_train_true - y_train_pred) ** 2)
    ss_tot_train = np.sum((y_train_true - np.mean(y_train_true)) ** 2)
    r2_train = 1 - ss_res_train / ss_tot_train

    ss_res_cv = np.sum((y_cv_true - y_cv_pred) ** 2)
    ss_tot_cv = np.sum((y_cv_true - np.mean(y_cv_true)) ** 2)
    r2_cv = 1 - ss_res_cv / ss_tot_cv

    print(f"R² (Training): {r2_train:.4f}")
    print(f"R² (CV): {r2_cv:.4f}")

    return W, b, cost_history_train, cost_history_cv, y_train_pred, y_cv_pred, feature_cols, fig


def main():
    """Run all visualizations."""
    print("Loading and preprocessing data...")
    df = load_and_preprocess_data()

    print(f"\nDataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # Plot all features vs revenue
    print("\nGenerating scatter plot grid for all features vs revenue...")
    df, fig1 = plot_features_vs_revenue(df)
    plt.show()

    # Plot correlation heatmap
    print("\nGenerating correlation heatmap...")
    corr_matrix = plot_correlation_heatmap(df)
    plt.show()

    # Plot categorical features
    print("\nGenerating categorical feature plots...")
    df, fig2 = plot_categorical_vs_revenue(df)
    plt.show()

    # Print summary statistics
    print("\n=== Feature Correlations with Revenue ===")
    p_features = [f"P{i}" for i in range(1, 38)]
    other_features = ["age", "CityGroup_Big_Cities", "Type_FC", "Type_IL"]
    all_features = p_features + other_features

    correlations = df[all_features + ["revenue"]].corr()["revenue"].drop("revenue").sort_values(
        key=abs, ascending=False
    )
    print(correlations.head(10))

    # Plot J_train vs J_cv using split data
    print("\n" + "="*60)
    print("Training Linear Regression and plotting J_train vs J_cv")
    print("="*60)
    W, b, cost_history_train, cost_history_cv, y_train_pred, y_cv_pred, feature_cols, fig3 = plot_j_train_vs_j_cv()
    plt.show()

    return df, corr_matrix, fig3


if __name__ == "__main__":
    main()