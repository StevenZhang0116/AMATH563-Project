import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


def load_and_prepare_data(filename):
    # Load data
    data = np.load(filename)
    labels = np.square(data)  # Squaring to create labels
    return data, labels


# Load and prepare training and testing data
data, labels = load_and_prepare_data("toy_quadratic_data.npy")
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)


def train_test(x_train_pca, x_test_pca, y_train, y_test, model):
    # Fit and predict
    model.fit(x_train_pca, y_train)
    pred = model.predict(x_test_pca)
    return pred


# PCA
max_pca_components = 120
pca = PCA(n_components=max_pca_components)
x_train_pca_full = pca.fit_transform(x_train)
x_test_pca_full = pca.transform(x_test)

# Cumulative Explained Variance
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.title("Cumulative Explained Variance")
plt.grid(True)
plt.show()

# Different numbers of PCA components
pca_modes = np.arange(1, max_pca_components + 1, 4)
errors = []
model = Ridge(alpha=0.1)
for n_pca in pca_modes:
    x_train_pca = x_train_pca_full[:, :n_pca]
    x_test_pca = x_test_pca_full[:, :n_pca]
    pred = train_test(x_train_pca, x_test_pca, y_train, y_test, model)
    error = np.mean(np.linalg.norm(pred - y_test, axis=-1) / np.linalg.norm(y_test, axis=-1))
    errors.append(error)

plt.figure()
plt.plot(pca_modes, errors, '-go')
plt.yscale('log')
plt.title("Error as a Function of PCA Modes")
plt.grid(True)
plt.show()

# Best number of PCA components
best_n_pca = pca_modes[np.argmin(errors)]
x_train_pca_best = x_train_pca_full[:, :best_n_pca]
x_test_pca_best = x_test_pca_full[:, :best_n_pca]
pred_best = train_test(x_train_pca_best, x_test_pca_best, y_train, y_test, model)
best_error = np.mean(np.linalg.norm(pred_best - y_test, axis=-1) / np.linalg.norm(y_test, axis=-1))
print(f"Best PCA Components: {best_n_pca}, Best Error: {best_error}")
