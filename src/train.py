import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle 
import yaml
import os
import mlflow
from mlflow.models import infer_signature
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV

def train(data_path, model_path, random_state, n_estimators, max_depth, mlflow_params):
    # Lecture locale (DVC s'occupe de garantir que le fichier est là)
    data = pd.read_csv(data_path)
    X = data.drop(columns=["Outcome"])
    y = data["Outcome"]

    mlflow.set_tracking_uri(mlflow_params["MLFLOW_TRACKING_URI"])
    
    with mlflow.start_run():
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

        # Grille de paramètres
        param_grid = {'min_samples_leaf': [1, 2]}

        mlflow.set_tag("model_type", "RandomForestClassifier")

        # Tuning
        rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=2, n_jobs=-1)
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy}")
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_params(grid_search.best_params_)

        # Sauvegarde locale du modèle (pour DVC)
        with open(model_path, "wb") as f:
            pickle.dump(best_model, f)

        # Logging MLflow
        signature = infer_signature(X_test, y_pred)
        mlflow.sklearn.log_model(best_model, "model", signature=signature)

if __name__ == "__main__":
    config = yaml.safe_load(open("params.yaml"))
    train_params = config['train']
    mlflow_params = config['mlflow']

    # On ne cherche plus aws_params dans le YAML. 
    # Boto3 et MLflow liront les clés depuis 'aws configure' automatiquement.
    
    train(
        train_params["data"], 
        train_params["model_path"], 
        train_params["random_state"], 
        train_params["n_estimators"], 
        train_params['max_depth'],
        mlflow_params
    )