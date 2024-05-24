import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.kernel_ridge import KernelRidge
import scipy.stats as stats
from sklearn.metrics import make_scorer


def load_and_prepare_data(filename):
    """Load data and create labels by squaring the data."""
    data = np.load(filename)
    labels = np.square(data)
    return data, labels


def relative_mse(y_true, y_pred):
    """Calculate the average relative MSE."""
    return np.mean(((y_pred - y_true) ** 2) / (y_true ** 2))


def mse_relative(y_true, y_pred):
    """
    :param y_true:
    :param y_pred:
    :return: average relative mse
    """
    # Handle y_true is zero
    non_zero_mask = y_true != 0
    return np.mean(np.square(np.linalg.norm(y_true[non_zero_mask] - y_pred[non_zero_mask], axis=-1) / np.linalg.norm(y_true[non_zero_mask], axis=-1)))

# Custom scorer for use in RandomizedSearchCV
relative_mse_scorer = make_scorer(relative_mse, greater_is_better=False)

iterationslst = [1000,2000,3000,4000,5000,6000,7000,8000,9000,10000]
rmselst = []

for iteration in iterationslst:
    # Load and prepare training and testing data
    data, labels = load_and_prepare_data(f"toy_quadratic_data_iter_{iteration}_order_10.npy")
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    # Scale the data
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # Apply PCA
    pca = PCA(n_components=0.9999)
    x_train_pca = pca.fit_transform(x_train_scaled)
    x_test_pca = pca.transform(x_test_scaled)

    # Calculate the median of pairwise distances for RBF kernel gamma
    median_dist = np.median(np.sqrt(np.sum(np.square(x_train_pca))))
    # print(median_dist)

    # Train fixed polynomial kernel
    poly_model = KernelRidge(kernel='poly', alpha=0.1, degree=2, coef0=1)
    poly_model.fit(x_train_pca, y_train)
    poly_pred = poly_model.predict(x_test_pca)

    # RBF model with RandomizedSearchCV using relative MSE scorer
    param_dist_rbf = {
        # 'gamma': stats.uniform((1/(0.1*median_dist)**2), 1/(0.01*median_dist)**2)
        #'gamma': 6.4e-5
    }
    # rbf_model = RandomizedSearchCV(KernelRidge(kernel='rbf', alpha=1), param_distributions=param_dist_rbf, n_iter=20, cv=10, scoring=relative_mse_scorer)
    rbf_model = KernelRidge(kernel='rbf', alpha=1, gamma=6.4e-5)
    rbf_model.fit(x_train_pca, y_train)
    rbf_pred = rbf_model.predict(x_test_pca)

    # Visualization
    # plt.figure(figsize=(12, 6))
    # plt.subplot(1, 2, 1)
    # plt.plot(y_test[5], label='Actual')
    # plt.plot(poly_pred[5], label='Poly Predicted', linestyle='dashed')
    # plt.title('Polynomial Kernel Prediction')
    # plt.legend()
    #
    # plt.subplot(1, 2, 2)
    # plt.plot(y_test[5], label='Actual')
    # plt.plot(rbf_pred[5], label='RBF Predicted', linestyle='dashed')
    # # plt.title(f'RBF Kernel Prediction\nBest Params: {rbf_model.best_params_}')
    # plt.title(f'RBF Kernel Prediction')
    # plt.legend()
    #
    # plt.tight_layout()
    # plt.show()

    # Print optimized parameters and errors for RBF kernel
    optimized_error = np.mean(np.linalg.norm(rbf_pred - y_test, axis=-1) / np.linalg.norm(y_test, axis=-1))
    # print(f"Optimized Parameters for RBF Kernel")
    # print(f"Optimized Error: {optimized_error}")

    rmse = mse_relative(y_test, rbf_pred)
    print(rmse)

    rmselst.append(rmse)

plt.figure()
plt.plot(iterationslst, rmselst)
result = [iterationslst, rmselst]
np.save('rbf_10.npy', result)

