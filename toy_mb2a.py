import numpy as np
from sklearn.decomposition import PCA
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import scipy.stats as stats


def load_and_prepare_data(filename):
    """Load data and create labels by squaring the data."""
    data = np.load(filename)
    labels = np.square(data)
    return data, labels


# Load and prepare training and testing data
data, labels = load_and_prepare_data("toy_quadratic_data_10000.npy")
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Apply PCA
max_pca_components = 950
pca = PCA(n_components=0.95)
x_train_pca_full = pca.fit_transform(x_train_scaled)
x_test_pca_full = pca.transform(x_test_scaled)

# Plot cumulative explained variance
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.title("Cumulative Explained Variance with PCA")
plt.grid(True)
plt.show()

def train_test_kernel_ridge(x_train, x_test, y_train, y_test, model):
    """Fit the Kernel Ridge model and predict on the test data."""
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    return pred

# Find the best kernel type
kernels = ['linear', 'poly', 'rbf']
errors = {}

for kernel in kernels:
    degree = 2 if kernel == 'poly' else 3  # Provide a default integer value for non-polynomial kernels
    model = KernelRidge(kernel=kernel, alpha=0.1, degree=degree)
    pred = train_test_kernel_ridge(x_train_pca_full, x_test_pca_full, y_train, y_test, model)
    error = np.mean(np.linalg.norm(pred - y_test, axis=-1) / np.linalg.norm(y_test, axis=-1))
    errors[kernel] = error

# Plot errors for different kernels
plt.bar(errors.keys(), errors.values(), color=['blue', 'green', 'red'])
plt.title("Error by Kernel Type with PCA")
plt.ylabel("Normalized Error")
plt.grid(True)
plt.show()

# Display the best kernel type
best_kernel = min(errors, key=errors.get)
print(f"Best Kernel: {best_kernel}, Best Error: {errors[best_kernel]}")

# Use RandomizedSearchCV for fine-tuning the best kernel
if best_kernel == 'linear':
    param_dist = {
        'alpha': stats.uniform(0.01, 10)
    }
elif best_kernel == 'poly':
    param_dist = {
        'alpha': stats.uniform(0.01, 10),
        'degree': stats.randint(2, 5),
        'coef0': stats.randint(0, 2)
    }
elif best_kernel == 'rbf':
    param_dist = {
        'alpha': stats.uniform(0.01, 10),
        'gamma': stats.uniform(0.01, 10)#reduce the length scale for rbf--->alpha
        #find median(u(x_i)) and then use alpha*median(U)
    }

# Perform random search
model_best = KernelRidge(kernel=best_kernel)
random_search = RandomizedSearchCV(model_best, param_distributions=param_dist, n_iter=20, cv=5, scoring='neg_mean_squared_error')
random_search.fit(x_train_pca_full, y_train)

# Get predictions and calculate the error for the optimized model
best_model = random_search.best_estimator_
pred = best_model.predict(x_test_pca_full)
optimized_error = np.mean(np.linalg.norm(pred - y_test, axis=-1) / np.linalg.norm(y_test, axis=-1))

# Print optimized parameters and error
print(f"Optimized Parameters for {best_kernel} Kernel: {random_search.best_params_}")
print(f"Optimized Error: {optimized_error}")

# Plot two random examples comparing predictions and actual values
np.random.seed(42)
random_indices = np.random.choice(range(len(x_test)), size=2, replace=False)

plt.figure(figsize=(10, 5))
for i, index in enumerate(random_indices):
    plt.subplot(1, 2, i + 1)
    plt.plot(y_test[index], label='Actual')
    plt.plot(pred[index], label='Predicted', linestyle='dashed')
    plt.title(f"Example {i + 1}")
    plt.legend()
plt.tight_layout()
plt.show()

#coef0 use 1