def train_model(model, X_train, y_train):
    # Special case: Isolation Forest is unsupervised
    if hasattr(model, 'fit_predict') and 'IsolationForest' in model.__class__.__name__:
        model.fit(X_train)
    else:
        model.fit(X_train, y_train)
    return model
