import pandas as pd
import numpy as np
from pathlib import Path

# Try to import sklearn for stratified splitting, fall back to random if not available
try:
    from sklearn.model_selection import train_test_split
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

import matplotlib.pyplot as plt
import seaborn as sns


def calculate_restaurant_age(open_date, reference_date="2026-01-01"):
    """
    Calculate the age of a restaurant in years.

    Parameters:
        open_date (str): Date string in MM/DD/YYYY format
        reference_date (str): Reference date in YYYY-MM-DD format (default: 2026-01-01)

    Returns:
        float: Age in years
    """
    # Parse the open date (format: MM/DD/YYYY)
    open_date_parsed = pd.to_datetime(open_date, format="%m/%d/%Y")
    # Parse reference date
    reference = pd.to_datetime(reference_date)
    # Calculate difference in years
    age_years = (reference - open_date_parsed).days / 365.25
    return max(0, age_years)  # Ensure non-negative


def analyze_data(filepath=None, reference_date="2026-01-01"):
    """Load and analyze the restaurant data."""
    if filepath is None:
        # Default to data/train.csv relative to this script's location
        filepath = Path(__file__).parent / "data" / "train.csv"

    df = pd.read_csv(filepath)

    # Calculate restaurant ages before dropping the columns
    print(f"\n--- Restaurant Age Calculation (reference date: {reference_date}) ---")
    df["age"] = df["Open Date"].apply(lambda x: calculate_restaurant_age(x, reference_date))

    # Display age info
    print(f"\nFirst 5 ages:")
    for i, age in enumerate(df["age"].head(5)):
        print(f"  {i+1}. {age:.1f} years")

    # Drop the Id, Open Date, City, City Group, and Type columns as they are not needed for analysis
    df = df.drop(columns=["Id", "Open Date", "City", "City Group", "Type"])

    # Display data statistics
    print(f"\n--- Data Statistics ---")
    print(f"Number of rows: {df.shape[0]}")
    print(f"Number of columns: {df.shape[1]}")
    print(f"\nColumn names and data types:")
    for col in df.columns:
        print(f"  {col}: {df[col].dtype}")
    print(f"\nColumn value ranges:")
    for col in df.columns:
        min_val = df[col].min()
        max_val = df[col].max()
        print(f"  {col}: [{min_val}, {max_val}]")

    # Revenue analysis
    revenue = df["revenue"]
    print(f"\n--- Revenue Analysis (in USD) ---")
    print(f"Mean: ${revenue.mean():,.2f}")
    print(f"Min:  ${revenue.min():,.2f}")
    print(f"Max:  ${revenue.max():,.2f}")
    print(f"Standard Deviation: ${revenue.std():,.2f}")

    # Multidimensional graph plotted against Y (revenue) with all features
    print(f"\n--- Generating multidimensional plot of features vs Revenue ---")
    plot_features_vs_revenue(df)

    # Display first 5 rows of the processed data
    print(f"\n--- First 5 Rows of Processed Data ---")
    print(df.head(5).to_string())

    return df


def plot_features_vs_revenue(df=None, filepath=None, reference_date="2026-01-01",
                              max_cols_per_row=5, figsize=None):
    """
    Create a grid of scatter plots showing all features against revenue.

    Parameters:
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
    other_features = ["age"]

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


def load_and_preprocess_data(filepath=None, reference_date="2026-01-01"):
    """
    Load and preprocess the restaurant data.

    Parameters:
        filepath : str or Path, optional
            Path to the CSV file. Defaults to data/train.csv relative to script.
        reference_date : str, optional
            Reference date for age calculation in YYYY-MM-DD format.
            Default is "2026-01-01".

    Returns:
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

    # Drop the Id, Open Date, City, City Group, and Type columns as they are not needed
    df = df.drop(columns=["Id", "Open Date", "City", "City Group", "Type"])

    return df


def split_and_normalize_data(df, target_col="revenue", train_ratio=0.6, cv_ratio=0.2, test_ratio=0.2, random_state=42):
    """
    Split data into training (60%), CV (20%), and test (20%) sets with stratification,
    then apply z-score normalization using training data statistics.

    Parameters:
        df : pd.DataFrame
            Input dataframe with features and target
        target_col : str
            Name of the target column
        train_ratio : float
            Proportion of data for training (default 0.6)
        cv_ratio : float
            Proportion of data for cross-validation (default 0.2)
        test_ratio : float
            Proportion of data for testing (default 0.2)
        random_state : int
            Random seed for reproducibility

    Returns:
        tuple: (train_df, cv_df, test_df, stats_dict)
            train_df, cv_df, test_df are normalized dataframes
            stats_dict contains mean and std for each feature from training data
    """
    # Define feature columns (all except the target)
    feature_cols = [col for col in df.columns if col != target_col]

    # Separate features and target
    X = df[feature_cols].values.copy()
    y = df[target_col].values.copy()

    if HAS_SKLEARN:
        # Convert target to bins for stratification based on revenue categories
        # Create 5 equal-sized bins for stratification
        bins = pd.qcut(y, q=5, labels=False, duplicates='drop')

        # First split: 60% train, 40% temp (CV + Test)
        X_train, X_temp, y_train, y_temp, _, bins_temp = train_test_split(
            X, y, bins, test_size=(1-train_ratio), random_state=random_state, stratify=bins
        )

        # Second split: Split temp into 20% CV and 20% test (50% each of temp)
        X_cv, X_test, y_cv, y_test, _, _ = train_test_split(
            X_temp, y_temp, bins_temp, test_size=0.5, random_state=random_state, stratify=bins_temp
        )
    else:
        # Simple random split without stratification
        np.random.seed(random_state)
        n_samples = len(df)
        indices = np.random.permutation(n_samples)

        train_end = int(n_samples * train_ratio)
        cv_end = int(n_samples * (train_ratio + cv_ratio))

        train_idx = indices[:train_end]
        cv_idx = indices[train_end:cv_end]
        test_idx = indices[cv_end:]

        X_train, y_train = X[train_idx], y[train_idx]
        X_cv, y_cv = X[cv_idx], y[cv_idx]
        X_test, y_test = X[test_idx], y[test_idx]

    # Calculate mean and std from training data for z-score normalization
    train_mean = np.mean(X_train, axis=0)
    train_std = np.std(X_train, axis=0)

    # Avoid division by zero for constant features
    train_std[train_std == 0] = 1.0

    # Apply z-score normalization: z = (x - mean) / std
    X_train_norm = (X_train - train_mean) / train_std
    X_cv_norm = (X_cv - train_mean) / train_std
    X_test_norm = (X_test - train_mean) / train_std

    # Create normalized dataframes
    train_df = pd.DataFrame(X_train_norm, columns=feature_cols)
    train_df[target_col] = y_train

    cv_df = pd.DataFrame(X_cv_norm, columns=feature_cols)
    cv_df[target_col] = y_cv

    test_df = pd.DataFrame(X_test_norm, columns=feature_cols)
    test_df[target_col] = y_test

    # Create stats dictionary
    stats_dict = {
        'mean': {col: train_mean[i] for i, col in enumerate(feature_cols)},
        'std': {col: train_std[i] for i, col in enumerate(feature_cols)},
        'feature_cols': feature_cols,
        'train_size': len(train_df),
        'cv_size': len(cv_df),
        'test_size': len(test_df)
    }

    return train_df, cv_df, test_df, stats_dict


def analyze_and_split_data(filepath=None, reference_date="2026-01-01"):
    """Load, analyze, split data into train/CV/test, and apply z-score normalization."""
    df = load_and_preprocess_data(filepath, reference_date)

    # Display data statistics
    print(f"\n--- Data Statistics ---")
    print(f"Number of rows: {df.shape[0]}")
    print(f"Number of columns: {df.shape[1]}")
    print(f"\nColumn names and data types:")
    for col in df.columns:
        print(f"  {col}: {df[col].dtype}")
    print(f"\nColumn value ranges:")
    for col in df.columns:
        min_val = df[col].min()
        max_val = df[col].max()
        print(f"  {col}: [{min_val}, {max_val}]")

    # Revenue analysis
    revenue = df["revenue"]
    print(f"\n--- Revenue Analysis (in USD) ---")
    print(f"Mean: ${revenue.mean():,.2f}")
    print(f"Min:  ${revenue.min():,.2f}")
    print(f"Max:  ${revenue.max():,.2f}")
    print(f"Standard Deviation: ${revenue.std():,.2f}")

    # Split and normalize data
    print(f"\n--- Splitting Data (60/20/20) and Z-score Normalization ---")
    train_df, cv_df, test_df, stats_dict = split_and_normalize_data(df, target_col="revenue")

    print(f"\nDataset Sizes:")
    print(f"  Training: {stats_dict['train_size']} rows ({stats_dict['train_size']/len(df)*100:.0f}%)")
    print(f"  CV: {stats_dict['cv_size']} rows ({stats_dict['cv_size']/len(df)*100:.0f}%)")
    print(f"  Test: {stats_dict['test_size']} rows ({stats_dict['test_size']/len(df)*100:.0f}%)")

    print(f"\nTraining Set Statistics (for normalization):")
    print(f"  Features: {len(stats_dict['feature_cols'])}")
    for col in stats_dict['feature_cols'][:5]:  # Show first 5
        print(f"  {col}: mean={stats_dict['mean'][col]:.4f}, std={stats_dict['std'][col]:.4f}")

    return train_df, cv_df, test_df, stats_dict


def save_splits_to_files(train_df, cv_df, test_df, output_dir=None):
    """
    Save the train, CV, and test dataframes to CSV files.

    Parameters:
        train_df : pd.DataFrame
            Training data
        cv_df : pd.DataFrame
            Cross-validation data
        test_df : pd.DataFrame
            Test data
        output_dir : str or Path, optional
            Directory to save files. Defaults to 'data/' subdirectory.
    """
    if output_dir is None:
        output_dir = Path(__file__).parent / "data"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(output_dir / "train_split.csv", index=False)
    cv_df.to_csv(output_dir / "cv_split.csv", index=False)
    test_df.to_csv(output_dir / "test_split.csv", index=False)

    print(f"\nSaved split datasets to:")
    print(f"  {output_dir / 'train_split.csv'}")
    print(f"  {output_dir / 'cv_split.csv'}")
    print(f"  {output_dir / 'test_split.csv'}")


def train_linear_regression(X, y_target, learning_rate=0.01, n_iterations=1000):
    """
    Train linear regression model using gradient descent.

    Parameters:
        X : np.ndarray, shape (n_samples, n_features)
            Input feature matrix
        y_target : np.ndarray, shape (n_samples,)
            Target values (revenue)
        learning_rate : float
            Learning rate for gradient descent
        n_iterations : int
            Number of iterations for gradient descent

    Returns:
        tuple: (W, b, cost_history, y_train)
            W: weights, shape (1, n_features)
            b: bias
            cost_history: list of cost values at each iteration
            y_train: predictions on training data
    """
    n_samples, n_features = X.shape

    # Initialize weights W as row vector (1 x n_features) and bias b as scalar
    W = np.zeros((1, n_features))
    b = 0.0

    # Reshape y_target to column vector for computation
    y_target_col = y_target.reshape(-1, 1)

    cost_history = []

    print(f"\n--- Training Linear Regression ---")
    print(f"Input X shape: {X.shape}")
    print(f"y_target shape: {y_target.shape}")
    print(f"W shape: {W.shape} (row vector)")
    print(f"b: {b} (scalar)")

    for i in range(n_iterations):
        # Forward pass: Y_train = WX + b
        # X is (n_samples, n_features), W is (1, n_features)
        # y_train = X @ W.T + b = W @ X.T + b
        y_train_col = (X @ W.T + b).reshape(-1, 1)  # Shape: (n_samples, 1)

        # Compute predictions as 1D array
        y_train = y_train_col.flatten()  # Shape: (n_samples,)

        # Compute cost: J(W,b) = 1/(2*m) * sum((y_target - y_train)^2)
        m = n_samples
        cost = (1 / (2 * m)) * np.sum((y_target_col - y_train_col) ** 2)
        cost_history.append(cost)

        # Gradient descent
        # dW = (1/m) * X.T @ (y_train - y_target)
        # db = (1/m) * sum(y_train - y_target)
        error = y_train_col - y_target_col  # Shape: (n_samples, 1)
        dW = (1 / m) * (X.T @ error)  # Shape: (n_features, 1)
        db = (1 / m) * np.sum(error)

        # Update weights and bias
        W = W - learning_rate * dW.T  # Shape: (1, n_features)
        b = b - learning_rate * db

        # Print progress every 100 iterations
        if (i + 1) % 100 == 0 or i == 0:
            print(f"Iteration {i+1}: Cost = {cost:,.2f}")

    return W, b, cost_history, y_train


def plot_actual_vs_predicted(y_train,y_target,  save_path=None):
    """
    Plot actual revenue vs predicted revenue.

    Parameters:
        y_target : np.ndarray
            Actual revenue values
        y_train : np.ndarray
            Predicted revenue values
        save_path : str, optional
            Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(y_target, y_train, alpha=0.6, c="steelblue", edgecolor="w", s=60)

    # Add perfect prediction line
    min_val = min(y_target.min(), y_train.min())
    max_val = max(y_target.max(), y_train.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

    plt.xlabel('Predicted Revenue (Y_train)', fontsize=12)
    plt.ylabel('Actual Revenue (Y_target)', fontsize=12)
    plt.title('Actual Revenue vs Predicted Revenue', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to: {save_path}")
    plt.close()


def plot_cost_history(cost_history, save_path=None):
    """
    Plot cost function over iterations.

    Parameters:
        cost_history : list
            List of cost values at each iteration
        save_path : str, optional
            Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(cost_history, linewidth=2, color='steelblue')
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Cost J(W,b)', fontsize=12)
    plt.title('Cost Function Over Iterations', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to: {save_path}")
    plt.close()


def run_linear_regression_analysis():
    """Run complete linear regression analysis on the training data."""
    from pathlib import Path

    # Load the split training data
    train_path = Path(__file__).parent / "data" / "train_split.csv"
    df = pd.read_csv(train_path)

    print(f"\n--- Linear Regression Analysis ---")
    print(f"Loaded training data from: {train_path}")
    print(f"Number of samples: {len(df)}")

    # Separate features and target
    # X includes all columns except 'revenue' (which is the target)
    if 'revenue' in df.columns:
        feature_cols = [col for col in df.columns if col != 'revenue']
        X = df[feature_cols].values
        y_target = df['revenue'].values
    else:
        raise ValueError("Revenue column not found in training data")

    # Display feature info
    print(f"\nFeatures used ({len(feature_cols)}): {feature_cols}")
    print(f"Target: revenue")

    # Training parameters
    learning_rate = 0.01
    n_iterations = 1000

    # Train the model
    W, b, cost_history, y_train = train_linear_regression(
        X, y_target, learning_rate=learning_rate, n_iterations=n_iterations
    )

    # Print final results
    print(f"\n--- Final Results ---")
    print(f"Final cost: {cost_history[-1]:,.2f}")
    print(f"Final bias b: {b:,.2f}")

    # Create plots directory
    plots_dir = Path(__file__).parent / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Plot actual vs predicted
    plot_actual_vs_predicted(y_target, y_train, save_path=str(plots_dir / "actual_vs_predicted.png"))

    # Plot cost history
    plot_cost_history(cost_history, save_path=str(plots_dir / "cost_over_iterations.png"))

    # Compute additional metrics
    mse = np.mean((y_target - y_train) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_target - y_train))
    r2 = 1 - np.sum((y_target - y_train) ** 2) / np.sum((y_target - np.mean(y_target)) ** 2)

    print(f"\n--- Model Performance ---")
    print(f"MSE: {mse:,.2f}")
    print(f"RMSE: {rmse:,.2f}")
    print(f"MAE: {mae:,.2f}")
    print(f"R²: {r2:.4f}")

    return W, b, cost_history, y_train, y_target


if __name__ == "__main__":
    # Run complete linear regression analysis
    run_linear_regression_analysis()