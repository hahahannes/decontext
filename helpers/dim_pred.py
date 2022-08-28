from sklearn.linear_model import ElasticNetCV, MultiTaskElasticNetCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import explained_variance_score, r2_score
import numpy as np 
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn import preprocessing
from scipy.spatial.distance import squareform
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from scipy.stats import spearmanr, pearsonr
import pandas as pd

def scale_data(X, y):
    X_scaler = preprocessing.StandardScaler()
    X_scaled = X_scaler.fit_transform(X)  # standardize activations (X)
    return X_scaled, y, X_scaler

def predict_outer_test(embeddings, model):
    preds_spose_dimensions = model.predict(embeddings) 
    preds_spose_dimensions[preds_spose_dimensions < 0] = 0  # set all negative predictions to 0 (because spose is positive)
    return preds_spose_dimensions 

def fit_model(embeddings, spose_dimensions, random_state):
    cv = RepeatedKFold(n_splits=5, n_repeats=1, random_state=random_state)
    #ratios = [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]
    ratios = [0.5]
    n_alphas = 1
    base = ElasticNetCV(l1_ratio=ratios, n_alphas=n_alphas, fit_intercept=False, cv=cv, random_state=random_state)
    clf = MultiOutputRegressor(base)
    clf.fit(embeddings, spose_dimensions)
    return clf

def get_r2_scores(preds_spose_dimensions, test_spose_dimensions):
    r2 = []
    for i in range(49):        
        r2.append(r2_score(test_spose_dimensions[:,  i], preds_spose_dimensions[:, i]))
    return r2

def get_correlation_of_similarity_based_on_pred_dims_global(preds_spose_dimensions, test_spose_dimensions):
    pred_sim_vector = get_similarity_vector(preds_spose_dimensions)
    true_sim_vector = get_similarity_vector(test_spose_dimensions)
    global_correlation = get_correlation_between_sim_vectors(pred_sim_vector, true_sim_vector)
    return global_correlation

def get_similarity_vector(spose_dimensions):
    similarity_matrx = cosine_similarity(spose_dimensions, spose_dimensions)
    sim_vector = squareform(similarity_matrx, force='tovector', checks=False)
    return sim_vector 

def get_correlation_between_sim_vectors(pred_sim_vector, true_sim_vector):
    return spearmanr(true_sim_vector, pred_sim_vector).correlation

def get_correlations_of_similarity_per_dim_based_on_pred_dimensions(preds_spose_dimensions, spose_dimensions):
    correlations_per_dim = []
    for i in range(49):
        single_dim_pred = preds_spose_dimensions[:, i].reshape(-1, 1)
        similarity_matrx_pred = euclidean_distances(single_dim_pred, single_dim_pred)
        dim_vector_pred = squareform(similarity_matrx_pred, force='tovector', checks=False)
        
        single_dim_true = spose_dimensions[:, i].reshape(-1, 1)
        similarity_matrx_true = euclidean_distances(single_dim_true, single_dim_true)
        dim_vector_true = squareform(similarity_matrx_true, force='tovector', checks=False)

        dim_correlation = get_correlation_between_sim_vectors(dim_vector_pred, dim_vector_true)
        correlations_per_dim.append(dim_correlation)

    return correlations_per_dim

def get_correlations_per_dim(preds_spose_dimensions, spose_dimensions):
    correlations_per_dim = []
    for i in range(49):
        single_dim_pred = preds_spose_dimensions[:, i].reshape(-1, 1)
        single_dim_true = spose_dimensions[:, i].reshape(-1, 1)
        dim_correlation = get_correlation_between_sim_vectors(single_dim_pred, single_dim_true)
        correlations_per_dim.append(dim_correlation)

    return correlations_per_dim

def fit_predict_cv(embeddings, spose_dimensions, cv_folds):
    print(f'Embeddings: {embeddings.shape} SPOSE: {spose_dimensions.shape}')
    random_state = 42
    n_dimensions = 49
    n_samples = embeddings.shape[0]

    outer_cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    cv_scores = []
    global_correlations = []
    sim_correlations_per_dims = []
    correlations_per_dims = []
    predicted_test_dimensions = np.zeros((n_samples, n_dimensions))

    # save ids for test prediction list
    things_ids = embeddings.index

    # DF to numpy
    embeddings = embeddings.to_numpy()

    for i_split, (outer_train_ind, outer_test_ind) in enumerate(outer_cv.split(range(n_samples))):
        print(f'Run split {i_split} of {cv_folds}')
        train_embeddings, train_spose, X_scaler = scale_data(embeddings[outer_train_ind, :], spose_dimensions[outer_train_ind, :])
        test_embeddings = X_scaler.transform(embeddings[outer_test_ind, :])

        train_embeddings = embeddings[outer_train_ind, :]
        train_spose = spose_dimensions[outer_train_ind, :]
        test_embeddings = embeddings[outer_test_ind, :]

        # unscaled Y as predictions will be unscaled too
        test_spose_dimensions = spose_dimensions[outer_test_ind, :]

        print('Fit model')
        model = fit_model(train_embeddings, train_spose, random_state)
        preds_spose_dimensions_unscaled = predict_outer_test(test_embeddings, model)
        
        # save test predictions
        predicted_test_dimensions[outer_test_ind] = preds_spose_dimensions_unscaled

        print('Get scores')
        r2_scores = get_r2_scores(preds_spose_dimensions_unscaled, test_spose_dimensions)
        cv_scores.append(r2_scores)

        global_correlation = get_correlation_of_similarity_based_on_pred_dims_global(preds_spose_dimensions_unscaled, test_spose_dimensions)
        global_correlations.append(global_correlation)

        sim_correlations_per_dim = get_correlations_of_similarity_per_dim_based_on_pred_dimensions(preds_spose_dimensions_unscaled, test_spose_dimensions)
        sim_correlations_per_dims.append(sim_correlations_per_dim)

        correlations_per_dim = get_correlations_per_dim(preds_spose_dimensions_unscaled, test_spose_dimensions)
        correlations_per_dims.append(correlations_per_dim)
        
    
    # create DF of predicted test dimensions
    predicted_test_dimensions_df = pd.DataFrame(predicted_test_dimensions)
    predicted_test_dimensions_df = predicted_test_dimensions_df.set_index(things_ids)

    return cv_scores, global_correlations, sim_correlations_per_dims, correlations_per_dims, predicted_test_dimensions_df