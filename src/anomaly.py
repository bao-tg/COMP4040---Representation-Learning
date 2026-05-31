import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

def detect_outliers_isolation_forest(embeddings, contamination=0.01, random_state=42):
    """
    Uses Isolation Forest to detect structural anomalies in the embedding space.
    
    Parameters:
    - embeddings: The high-dimensional text representations.
    - contamination: The proportion of outliers in the data set (e.g., 0.01 for 1%).
    
    Returns:
    - is_anomaly: Boolean array where True means the review is an anomaly.
    - scores: Anomaly scores (lower means more anomalous).
    """
    print(f"Fitting Isolation Forest (contamination={contamination})...")
    
    # Subsample if dataset is absolutely massive to save memory/time, though IsolationForest
    # is usually fine up to ~1M samples. We'll use the default max_samples='auto'.
    clf = IsolationForest(
        contamination=contamination, 
        random_state=random_state, 
        n_jobs=-1
    )
    
    # Predict returns -1 for outliers and 1 for inliers
    preds = clf.fit_predict(embeddings)
    is_anomaly = (preds == -1)
    
    # decision_function returns average anomaly score of X of the base classifiers.
    # The anomaly score of an input sample is computed as the mean anomaly score of the trees in the forest.
    # The measure of normality of an observation given a tree is the depth of the leaf containing this observation.
    scores = clf.decision_function(embeddings)
    
    return is_anomaly, scores

def detect_rating_inconsistencies(embeddings, ratings, top_percent=0.01, random_state=42):
    """
    Detects reviews where the text sentiment conflicts with the star rating.
    It does this by training a Ridge Regression model to predict the rating from the embeddings.
    The reviews with the largest absolute prediction error (residuals) are flagged.
    
    Parameters:
    - embeddings: The text representations.
    - ratings: The actual star ratings (1-5).
    - top_percent: The percentage of reviews to flag as inconsistent.
    
    Returns:
    - is_inconsistent: Boolean array flagging the top inconsistent reviews.
    - residuals: The absolute difference between predicted and actual ratings.
    - predicted_ratings: The raw predicted rating values.
    """
    print("Training Ridge model to predict ratings from embeddings...")
    
    # We use Ridge regression because it handles collinearity well and is fast for dense embeddings
    model = Ridge(alpha=1.0, random_state=random_state)
    
    # For a robust approach, we could use cross-validation, but for simplicity and speed
    # on large full-run artifacts, fitting on the whole dataset to find residuals is acceptable for anomaly detection.
    # We want to find points that the model *cannot* fit well, even when seeing them in training.
    model.fit(embeddings, ratings)
    
    predicted_ratings = model.predict(embeddings)
    
    # Calculate absolute residuals
    residuals = np.abs(ratings - predicted_ratings)
    
    # Determine the threshold for the top N% residuals
    n_flag = max(1, int(len(ratings) * top_percent))
    
    # Find the threshold value
    threshold = np.partition(residuals, -n_flag)[-n_flag]
    
    # Flag the top inconsistencies
    is_inconsistent = residuals >= threshold
    
    return is_inconsistent, residuals, predicted_ratings
