from joblib import Parallel, delayed
from tqdm import tqdm
import sys, os
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.getcwd())
import numpy as np
import pandas as pd
from pyfrechet.metric_spaces import MetricData, Sphere, AnisotropicSphere, two_euclidean
from pyfrechet.metrics import mse
from sklearn.model_selection import train_test_split
import itertools
from contextlib import contextmanager
import joblib
from sklearn.metrics import mean_squared_error
from geomstats.learning.kmeans import RiemannianKMeans
import contextlib
from pyfrechet.regression.bagged_regressor import BaggedRegressor
from pyfrechet.regression.d_trees import d_Tree
from pyfrechet.regression.trees import Tree
from sklearn.metrics import make_scorer
import time
from sklearn.model_selection import KFold
import itertools
from pyfrechet.metric_spaces.utils import sq_D_mat  

np.random.seed(1000)
sign_level = np.array([0.01, 0.05, 0.1])

neg_mse = make_scorer(mse, greater_is_better=False) 

def ani_ani_custom_GCV(X_train, y_train, param_grid, seed=5, n_splits=5):
    """
    Manual Grid Search CV for Fréchet forests. Use the same lambda_ for preductors
    
    Parameters:
    - M: Metric space object
    - X_train: array-like, shape (n_samples, n_features)
    - y_train: array-like, shape (n_samples, 3)
    - param_grid: dict with keys 'min_split_size', 'mtry', 'lambda_'
    - seed: int, random seed for reproducibility
    - n_splits: int, number of CV folds

    Returns:
    - final_forest: fitted BaggedRegressor on full training data with best parameters
    - best: dict, best hyperparameters
    - cv_results: list of dicts with CV performance
    """
    grid = list(itertools.product(param_grid['min_split_size'],
                                  param_grid['mtry'],
                                  param_grid['c_lambda_']))

    cv_results = []
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for min_split_size, mtry, c_lambda_ in grid:
        fold_errors = []

        for train_index, val_index in kf.split(X_train):
            X_tr, X_val = X_train[train_index], X_train[val_index]
            y_tr_raw, y_val_raw = y_train[train_index], y_train[val_index]

            # Define metric and structure
            M = AnisotropicSphere(dim=2, c_lambda_=c_lambda_)
            # In structure, do not normalize the distance. This is done to compare the MSEs, so no need to apply it for the predictors. It would not change the results anyway.
            structure = [(AnisotropicSphere(dim=2, c_lambda_= (1, c_lambda_[1])), list(range(0, 3)))]
            y_tr = MetricData(M, y_tr_raw.reshape(-1, 3))
            y_val = MetricData(M, y_val_raw.reshape(-1, 3))

            # Define forest
            base = d_Tree(split_type='2means', impurity_method='medoid', structure=structure,
                        min_split_size=min_split_size, mtry=mtry)
            forest = BaggedRegressor(estimator=base, n_estimators=200,
                                     bootstrap_fraction=1, bootstrap_replace=True,
                                     seed=seed, n_jobs=10)

            forest.fit(X_tr, y_tr)
            preds = forest.predict(X_val)
            oob_quantile = np.percentile(forest.oob_errors(), 0.9 * 100, method='inverted_cdf')
            error = mse(y_val, preds) + 0.1*area_pred_ball(M, oob_quantile, 1e5)
            fold_errors.append(error)

        avg_error = np.mean(fold_errors)
        cv_results.append({
            'min_split_size': min_split_size,
            'mtry': mtry,
            'c_lambda_': c_lambda_,
            'cv_error': avg_error
        })
        print(f"Params: min_split_size={min_split_size}, mtry={mtry}, c_lambda_={c_lambda_}, CV error={avg_error:.4f}")

    best = min(cv_results, key=lambda x: x['cv_error'])
    print("\nBest parameters:")
    print(best)

    # Final refit on full training data
    best_M = AnisotropicSphere(dim=2, c_lambda_=best['c_lambda_'])
    best_structure = [(AnisotropicSphere(dim=2, c_lambda_=(1, best['c_lambda_'][1])), list(range(0, 3)))]
    y_train_metric = MetricData(best_M, y_train.reshape(-1, 3))
    final_tree = d_Tree(split_type='2means', impurity_method='medoid',
                      structure=best_structure,
                      min_split_size=best['min_split_size'], mtry=best['mtry'])
    final_forest = BaggedRegressor(estimator=final_tree, n_estimators=200,
                                   bootstrap_fraction=1, bootstrap_replace=True,
                                   seed=seed, n_jobs=10)
    final_forest.fit(X_train, y_train_metric)

    return final_forest, best, cv_results

def iso_ani_custom_GCV(X_train, y_train, param_grid, seed=5, n_splits=5):
    """
    Manual Grid Search CV for Fréchet forests. Use the same lambda_ for predictors
    
    Parameters:
    - M: Metric space object
    - X_train: array-like, shape (n_samples, n_features)
    - y_train: array-like, shape (n_samples, 3)
    - param_grid: dict with keys 'min_split_size', 'mtry', 'lambda_'
    - seed: int, random seed for reproducibility
    - n_splits: int, number of CV folds

    Returns:
    - final_forest: fitted BaggedRegressor on full training data with best parameters
    - best: dict, best hyperparameters
    - cv_results: list of dicts with CV performance
    """
    grid = list(itertools.product(param_grid['min_split_size'],
                                  param_grid['mtry'],
                                  param_grid['c_lambda_']))

    cv_results = []
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for min_split_size, mtry, c_lambda_ in grid:
        fold_errors = []

        for train_index, val_index in kf.split(X_train):
            X_tr, X_val = X_train[train_index], X_train[val_index]
            y_tr_raw, y_val_raw = y_train[train_index], y_train[val_index]

            # Define metric and structure
            M = AnisotropicSphere(dim=2, c_lambda_=c_lambda_)
            y_tr = MetricData(M, y_tr_raw.reshape(-1, 3))
            y_val = MetricData(M, y_val_raw.reshape(-1, 3))
            structure = [(Sphere(dim=2, c_=1), list(range(0, 3)))]
            
            # Define forest
            base = d_Tree(split_type='2means', impurity_method='medoid', structure=structure,
                        min_split_size=min_split_size, mtry=mtry)
            forest = BaggedRegressor(estimator=base, n_estimators=200,
                                     bootstrap_fraction=1, bootstrap_replace=True,
                                     seed=seed, n_jobs=10)

            forest.fit(X_tr, y_tr)
            preds = forest.predict(X_val)
            oob_quantile = np.percentile(forest.oob_errors(), 0.9 * 100, method='inverted_cdf')
            error = mse(y_val, preds) + 0.1*area_pred_ball(M, oob_quantile, 1e5)
            fold_errors.append(error)

        avg_error = np.mean(fold_errors)
        cv_results.append({
            'min_split_size': min_split_size,
            'mtry': mtry,
            'c_lambda_': c_lambda_,
            'cv_error': avg_error
        })
        print(f"Params: min_split_size={min_split_size}, mtry={mtry}, c_lambda_={c_lambda_}, CV error={avg_error:.4f}")

    best = min(cv_results, key=lambda x: x['cv_error'])
    print("\nBest parameters:")
    print(best)

    # Final refit on full training data
    best_M = AnisotropicSphere(dim=2, c_lambda_=best['c_lambda_'])
    y_train_metric = MetricData(best_M, y_train.reshape(-1, 3))
    final_tree = d_Tree(split_type='2means', impurity_method='medoid',
                      structure=structure,
                      min_split_size=best['min_split_size'], mtry=best['mtry'])
    final_forest = BaggedRegressor(estimator=final_tree, n_estimators=200,
                                   bootstrap_fraction=1, bootstrap_replace=True,
                                   seed=seed, n_jobs=10)
    final_forest.fit(X_train, y_train_metric)

    return final_forest, best, cv_results

def custom_GCV(M_response, structure, X_train, y_train, param_grid, seed=5, n_splits=5):
    """
    Manual Grid Search CV for Fréchet forests.
    
    Parameters:
    - M: Metric space object
    - X_train: array-like, shape (n_samples, n_features)
    - y_train: array-like, shape (n_samples, 3)
    - param_grid: dict with keys 'min_split_size', 'mtry', 'lambda_'
    - seed: int, random seed for reproducibility
    - n_splits: int, number of CV folds

    Returns:
    - final_forest: fitted BaggedRegressor on full training data with best parameters
    - best: dict, best hyperparameters
    - cv_results: list of dicts with CV performance
    """
    grid = list(itertools.product(param_grid['min_split_size'],
                                  param_grid['mtry'],
                                  param_grid['c_']))

    cv_results = []
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for min_split_size, mtry, c_ in grid:
        fold_errors = []

        for train_index, val_index in kf.split(X_train):
            X_tr, X_val = X_train[train_index], X_train[val_index]
            y_tr_raw, y_val_raw = y_train[train_index], y_train[val_index]

            # Define metric and structure
            y_tr = MetricData(M_response, y_tr_raw.reshape(-1, 3))
            y_val = MetricData(M_response, y_val_raw.reshape(-1, 3))

            # Define forest
            base = d_Tree(split_type='2means', impurity_method='medoid', structure=structure,
                        min_split_size=min_split_size, mtry=mtry)
            forest = BaggedRegressor(estimator=base, n_estimators=200,
                                     bootstrap_fraction=1, bootstrap_replace=True,
                                     seed=seed, n_jobs=10)

            forest.fit(X_tr, y_tr)
            preds = forest.predict(X_val)
            oob_quantile = np.percentile(forest.oob_errors(), 0.9 * 100, method='inverted_cdf')
            error = mse(y_val, preds)
            fold_errors.append(error)

        avg_error = np.mean(fold_errors)
        cv_results.append({
            'min_split_size': min_split_size,
            'mtry': mtry,
            'c': c_,
            'cv_error': avg_error
        })
        print(f"Params: min_split_size={min_split_size}, mtry={mtry}, c_={c_}, CV error={avg_error:.4f}")

    best = min(cv_results, key=lambda x: x['cv_error'])
    print("\nBest parameters:")
    print(best)

    # Final refit on full training data
    y_train_metric = MetricData(M_response, y_train.reshape(-1, 3))
    final_tree = d_Tree(split_type='2means', impurity_method='medoid',
                      structure=structure,
                      min_split_size=best['min_split_size'], mtry=best['mtry'])
    final_forest = BaggedRegressor(estimator=final_tree, n_estimators=200,
                                   bootstrap_fraction=1, bootstrap_replace=True,
                                   seed=seed, n_jobs=10)
    final_forest.fit(X_train, y_train_metric)

    return final_forest, best, cv_results

def area_pred_ball(M, radius, total_points):
    """
    Estimate the area of a prediction ball using the distance of M with given radius.
    The area is calculated by sampling points "uniformly" on the unit sphere and checking if they are within the radius.
    
    Parameters:
    - M: Metric space
    - radius: float, radius of the ball
    - total_points: int, number of points to sample

    Returns:
    - area: float, area of the ball
    """
    x, y, z = canonical_lattice(total_points)
    return 4*np.pi*np.sum((M.d(np.vstack((x,y,z)).T, np.array([np.sqrt(2)/2,np.sqrt(2)/2,0])) < radius))/total_points


def canonical_lattice(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a canonical lattice on the unit sphere.
    
    Parameters:
    - n: int, number of points to generate

    Returns:
    - x, y, z: coordinates of the points on the sphere
    """
    goldenRatio = (1 + 5**0.5)/2
    i = np.arange(0, n)
    theta = 2 * np.pi * i / goldenRatio
    phi = np.arccos(1 - 2*(i+0.5)/n)
    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(phi)
    return x, y, z

@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def task(cyc):
    filename = f'sunspots_births_{cyc}_deaths.csv'
    print(f'Processing {filename}...')
    filepath = os.path.join(os.getcwd(), 'sunspots/data', filename)

    sample = pd.read_csv(filepath)

    X = np.vstack([sample['births_X.1'], sample['births_X.2'], sample['births_X.3']]).T
    y = np.vstack([sample['deaths_X.1'], sample['deaths_X.2'], sample['deaths_X.3']]).T
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1000)

    # Compute c_lambda_ values
    c_lambda_ = []
    for lambda_ in [0.5, 1, 1.5, 2, 3]:
        M = AnisotropicSphere(dim=2, c_lambda_=(1, lambda_))
        c_lambda_.append((1/np.mean(sq_D_mat(M, y_train)), lambda_))

    # ANISOTROPIC RESPONSE - ANISOTROPIC PREDICTORS
    param_grid = {'min_split_size': [1, 5, 10], 'mtry': [1], 'c_lambda_': c_lambda_}
    start = time.time()
    ani_ani_final_forest, ani_ani_best_params, _ = ani_ani_custom_GCV(X_train, y_train, param_grid)
    end = time.time()
    elapsed_minutes = (end - start) / 60
    print(f"Anisotropic response and predictors, cycle {cyc} time: {elapsed_minutes:.1f} minutes")
    M = AnisotropicSphere(dim=2, c_lambda_=ani_ani_best_params['c_lambda_'])
    ani_ani_oob_quantile = np.percentile(ani_ani_final_forest.oob_errors(), (1 - np.array([0.01, 0.05, 0.1])) * 100, method='inverted_cdf')
    ani_ani_preds = ani_ani_final_forest.predict(X_test)
    ani_ani_pb_ii_cov = np.sum(M.d(MetricData(M, y_test), ani_ani_preds).reshape(X_test.shape[0],1) <= np.tile(ani_ani_oob_quantile, X_test.shape[0]).reshape(-1,3), axis = 0) / X_test.shape[0]


    # ANISOTROPIC RESPONSE - ISOTROPIC PREDICTORS
    start = time.time()
    iso_ani_final_forest, iso_ani_best_params, _ = iso_ani_custom_GCV(X_train, y_train, param_grid)
    end = time.time()
    elapsed_minutes = (end - start) / 60
    print(f"Anisotropic response and isotropic predictors, cycle {cyc} time: {elapsed_minutes:.1f} minutes")
    M = AnisotropicSphere(dim=2, c_lambda_=iso_ani_best_params['c_lambda_'])
    iso_ani_oob_quantile = np.percentile(iso_ani_final_forest.oob_errors(), (1 - np.array([0.01, 0.05, 0.1])) * 100, method='inverted_cdf')
    iso_ani_preds = iso_ani_final_forest.predict(X_test)
    iso_ani_pb_ii_cov = np.sum(M.d(MetricData(M, y_test), iso_ani_preds).reshape(X_test.shape[0],1) <= np.tile(iso_ani_oob_quantile, X_test.shape[0]).reshape(-1,3), axis = 0) / X_test.shape[0]


    # ISOTROPIC RESPONSE - ISOTROPIC PREDICTORS
    M = Sphere(dim=2, c_=1)
    c_sphere_ = 1/np.mean(sq_D_mat(M, y_train))
    param_grid_iso = {'min_split_size': [1, 5, 10], 'mtry': [1], 'c_': [c_sphere_]}
    start = time.time()
    iso_forest, _, _ = custom_GCV(M_response=Sphere(dim=2, c_=c_sphere_), structure=[(Sphere(dim=2, c_=1), list(range(3)))], X_train=X_train, y_train=y_train, param_grid=param_grid_iso)
    end = time.time()
    elapsed_minutes = (end - start) / 60
    print(f"Isotropic response and predictors, cycle {cyc} time: {elapsed_minutes:.1f} minutes")
    iso_oob_quantile = np.percentile(iso_forest.oob_errors(), (1 - np.array([0.01, 0.05, 0.1])) * 100, method='inverted_cdf')
    iso_preds = iso_forest.predict(X_test)
    iso_pb_ii_cov = np.sum(M.d(MetricData(M, y_test), iso_preds).reshape(X_test.shape[0],1) <= np.tile(iso_oob_quantile, X_test.shape[0]).reshape(-1,3), axis = 0) / X_test.shape[0]


    # ISOTROPIC RESPONSE - EUCLIDEAN PREDICTORS
    X_train_euc = X_train.copy()
    X_test_euc = X_test.copy()
    start = time.time()
    euc_forest, _, _ = custom_GCV(M_response=Sphere(dim=2, c_=c_sphere_), structure=[(two_euclidean(1), [0]), (two_euclidean(1), [1])], X_train=X_train_euc, y_train=y_train, param_grid=param_grid_iso)
    end = time.time()
    elapsed_minutes = (end - start) / 60
    print(f"Isotropic response and Euclidean predictors, cycle {cyc} time: {elapsed_minutes:.1f} minutes")
    euc_oob_quantile = np.percentile(euc_forest.oob_errors(), (1 - np.array([0.01, 0.05, 0.1])) * 100, method='inverted_cdf')
    euc_preds = euc_forest.predict(X_test_euc)
    euc_pb_ii_cov = np.sum(M.d(MetricData(M, y_test), euc_preds).reshape(X_test_euc.shape[0],1) <= np.tile(euc_oob_quantile, X_test_euc.shape[0]).reshape(-1,3), axis = 0) / X_test_euc.shape[0]

    results = {}

    # Save the best parameters
    results['ani_ani_best_params'] = ani_ani_best_params
    results['iso_ani_best_params'] = iso_ani_best_params
    results['c_sphere_'] = c_sphere_

    # MSEs
    results['mse_ani_ani_geo'] = mse(MetricData(Sphere(2, c_=c_sphere_), y_test), MetricData(Sphere(2, c_=c_sphere_), ani_ani_preds.data))
    results['mse_ani_ani_anichord'] = mse(MetricData(AnisotropicSphere(2, ani_ani_best_params['c_lambda_']), y_test), MetricData(AnisotropicSphere(2, ani_ani_best_params['c_lambda_']), ani_ani_preds.data))
    results['mse_iso_ani_geo'] = mse(MetricData(Sphere(2, c_=c_sphere_), y_test), MetricData(Sphere(2, c_=c_sphere_), iso_ani_preds.data))
    results['mse_iso_ani_anichord'] = mse(MetricData(AnisotropicSphere(2, iso_ani_best_params['c_lambda_']), y_test), MetricData(AnisotropicSphere(2, iso_ani_best_params['c_lambda_']), iso_ani_preds.data))
    results['mse_iso_iso_geo'] = mse(MetricData(Sphere(2, c_=c_sphere_), y_test), MetricData(Sphere(2, c_=c_sphere_), iso_preds.data))
    results['mse_iso_iso_anichord'] = mse(MetricData(AnisotropicSphere(2, ani_ani_best_params['c_lambda_']), y_test), MetricData(AnisotropicSphere(2, ani_ani_best_params['c_lambda_']), iso_preds.data))
    results['mse_iso_euc_geo'] = mse(MetricData(Sphere(2, c_=c_sphere_), y_test), MetricData(Sphere(2, c_=c_sphere_), euc_preds.data))
    results['mse_iso_euc_anichord'] = mse(MetricData(AnisotropicSphere(2, ani_ani_best_params['c_lambda_']), y_test), MetricData(AnisotropicSphere(2, ani_ani_best_params['c_lambda_']), euc_preds.data))

    # OOB quantiles
    results['oob_quantile_ani_ani'] = ani_ani_oob_quantile
    results['oob_quantile_iso_ani'] = iso_ani_oob_quantile
    results['oob_quantile_iso_iso'] = iso_oob_quantile
    results['oob_quantile_iso_euc'] = euc_oob_quantile

    # PB coverages, Type II 
    results['pb_ii_cov_ani_ani'] = ani_ani_pb_ii_cov
    results['pb_ii_cov_iso_ani'] = iso_ani_pb_ii_cov
    results['pb_ii_cov_iso_iso'] = iso_pb_ii_cov
    results['pb_ii_cov_iso_euc'] = euc_pb_ii_cov

    # Areas
    M = AnisotropicSphere(2, ani_ani_best_params['c_lambda_'])
    results['area_ani_ani'] = [area_pred_ball(M, r, 100000) for r in ani_ani_oob_quantile]
    M = AnisotropicSphere(2, iso_ani_best_params['c_lambda_'])
    results['area_iso_ani'] = [area_pred_ball(M, r, 100000) for r in iso_ani_oob_quantile]
    M = Sphere(2, c_sphere_)
    results['area_iso_iso'] = [area_pred_ball(M, r, 100000) for r in iso_oob_quantile]
    results['area_iso_euc'] = [area_pred_ball(M, r, 100000) for r in euc_oob_quantile]


    # Save
    output_path = f'sunspots/results/results_cycle_{cyc}.npy'
    np.save(output_path, results)
    print("Sample saved to", output_path)


file_list = list(range(11, 25))  # cyc=11 to 24
total_files = len(file_list)


# with tqdm_joblib(tqdm(total=total_files)) as progress_bar:
#     Parallel(n_jobs=10)(delayed(task)(cyc) for cyc in file_list)


blocks = [
    [23],
    [21],
    [22],
    [17, 12],
    [16, 13],
    [15, 14],
    [18],
    [19],
    [20]
]


def process_block(block):
    # Process ONE block in parallel using 10 cores
    for cyc in block:
        task(cyc)

for block in blocks:
    print(f"Processing block: {block}")
    process_block(block)
