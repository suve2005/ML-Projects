import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv("PRSA_data.csv")
print(df.shape)
print(df.head())
print(df.columns)
print(df.dtypes)
print(df.isnull().sum()) # count of NA on each column

# cleaning the data
df=df.dropna(subset=['pm2.5'])
print(df.shape)

df=df.drop(columns=['No'])
print(df.shape)

print(df['cbwd'].unique())
print(df['cbwd'].value_counts())

# one-hot encode cbwd, drop NE as reference category
df = pd.get_dummies(df, columns=['cbwd'], drop_first=False)
df = df.drop(columns=['cbwd_NE'])   # NE is the reference

print(df.columns)
print(df.shape)
print(df.head())

df['cbwd_NW'] = df['cbwd_NW'].astype(int)
df['cbwd_SE'] = df['cbwd_SE'].astype(int)
df['cbwd_cv'] = df['cbwd_cv'].astype(int)
print(df.head())

# sort chronologically first — absolutely critical
# lag features only make sense if data is in time order
#take a possibly-out-of-order time series, put it into strict chronological order, and give it a clean sequential index that matches that order.
df = df.sort_values(['year', 'month', 'day', 'hour']).reset_index(drop=True)

# lag_1: pm2.5 exactly one hour ago
df['lag_1'] = df['pm2.5'].shift(1)

# lag_24: pm2.5 same hour yesterday
df['lag_24'] = df['pm2.5'].shift(24)

# roll_24: 24hr rolling mean — shift(1) FIRST to avoid leakage
df['roll_24'] = df['pm2.5'].shift(1).rolling(24).mean()

# roll_168: 1 week rolling mean — shift(1) FIRST to avoid leakage
df['roll_168'] = df['pm2.5'].shift(1).rolling(168).mean()

# drop rows where any lag/rolling feature is NaN
df = df.dropna()

# check result
print(df.shape)
print(df[['pm2.5', 'lag_1', 'lag_24', 'roll_24', 'roll_168']].head(10))

print(df.columns.tolist())   # all 18 column names
print(df.shape)              # confirm (41589, 18)
print(df.describe())         # statistics for every column — min, max, mean, std

# separate features X and target y
X = df.drop(columns=['pm2.5'])
y = df['pm2.5']

# chronological 80/20 split
split = int(len(df) * 0.8)

X_train = X.iloc[:split]
X_test  = X.iloc[split:]
y_train = y.iloc[:split]
y_test  = y.iloc[split:]

print(f"Train size: {X_train.shape}")
print(f"Test size:  {X_test.shape}")
print(f"Train period: {df['year'].iloc[0]} to {df['year'].iloc[split]}")
print(f"Test period:  {df['year'].iloc[split]} to {df['year'].iloc[-1]}")

# compute mean and std from TRAINING data only
mu    = X_train.mean()
sigma = X_train.std()

# normalize both train and test using TRAINING statistics
X_train_norm = (X_train - mu) / sigma
X_test_norm  = (X_test  - mu) / sigma

print(X_train_norm.describe().round(2))

# convert pandas DataFrames to numpy arrays for your model
X_train_np = X_train_norm.values    # shape (33271, 17)
X_test_np  = X_test_norm.values     # shape (8318, 17)
y_train_np = y_train.values         # shape (33271,)
y_test_np  = y_test.values          # shape (8318,)

print(X_train_np.shape)
print(y_train_np.shape)

import copy
import numpy as np

def compute_cost(X, y, w, b):
    m = X.shape[0]
    cost = 0.0
    for i in range(m):
        f_wb_i = np.dot(X[i], w) + b
        cost += (f_wb_i - y[i]) ** 2
    cost = cost / (2 * m)
    return cost
def compute_gradient_vectorized(X, y, w, b):
    m = X.shape[0]
    err = np.dot(X, w) + b - y        # shape (m,) — all errors at once
    dj_dw = np.dot(X.T, err) / m      # shape (n,) — all feature gradients at once
    dj_db = np.sum(err) / m           # scalar
    return dj_dw, dj_db

def compute_cost_vectorized(X, y, w, b):
    m = X.shape[0]
    err = np.dot(X, w) + b - y        # shape (m,)
    cost = np.dot(err, err) / (2 * m) # sum of squared errors
    return cost

def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters):
    J_hist = []
    w = copy.deepcopy(w_in)
    b = b_in
    for i in range(num_iters):
        dj_dw, dj_db = gradient_function(X, y, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        if i < 1000000:
            J_hist.append(cost_function(X, y, w, b))
        if i % 100 == 0:
            print(f"Iteration {i:4d}: Cost {J_hist[-1]:.2f}")
    return w, b, J_hist

initial_w = np.zeros(X_train_np.shape[1])
initial_b = 0
alpha      = 0.01
iterations = 1000

w_final, b_final, J_hist = gradient_descent(
    X_train_np, y_train_np,
    initial_w, initial_b,
    compute_cost_vectorized, compute_gradient_vectorized,
    alpha, iterations
)

print(f"Final b: {b_final:.4f}")
print(f"Final w: {w_final}")

# predict on test set
y_pred = np.dot(X_test_np, w_final) + b_final

# calculate MAE, RMSE, R²
mae  = np.mean(np.abs(y_pred - y_test_np))
rmse = np.sqrt(np.mean((y_pred - y_test_np)**2))
ss_res = np.sum((y_test_np - y_pred)**2)
ss_tot = np.sum((y_test_np - np.mean(y_test_np))**2)
r2   = 1 - (ss_res / ss_tot)

print(f"MAE:  {mae:.2f}  µg/m³")
print(f"RMSE: {rmse:.2f} µg/m³")
print(f"R²:   {r2:.4f}")

# plot predicted vs actual
plt.figure(figsize=(12, 4))
plt.plot(y_test_np[:200], label='Actual',    c='blue', alpha=0.7)
plt.plot(y_pred[:200],    label='Predicted', c='red',  alpha=0.7)
plt.title("Actual vs Predicted PM2.5 (first 200 test hours)")
plt.xlabel("Hour")
plt.ylabel("PM2.5 µg/m³")
plt.legend()
plt.show()
