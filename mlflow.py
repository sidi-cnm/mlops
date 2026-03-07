import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from mlflow.models import infer_signature

# 1. Correction de l'URI (suppression du point-virgule à la fin)
mlflow.set_tracking_uri("http://15.236.42.9:5000")

# Chargement des données
X, y = load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.9, random_state=123
)

# 2. Définition de l'expérience
mlflow.set_experiment("Logistic regression for Iris Data")

with mlflow.start_run():
    params = {
        "solver": "newton-cg",
        "max_iter": 100
    }

    # Entraînement
    llr = LogisticRegression(**params)
    llr.fit(X_train, y_train)

    # Prédictions
    y_pred_train = llr.predict(X_train)
    y_pred_test = llr.predict(X_test)

    # Calcul des métriques
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    score_f1 = f1_score(y_test, y_pred_test, average="macro")

    # 3. Log des paramètres et métriques
    mlflow.log_params(params)
    mlflow.log_metrics({
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "score_f1": score_f1
    })

    # Signature du modèle (schéma des données)
    llr_signature = infer_signature(X_train, llr.predict(X_train))

    # 4. Log du modèle dans S3
    # Note: Le nom du chemin (artifact_path) ne doit pas contenir d'espaces de préférence
    mlflow.sklearn.log_model(
        sk_model=llr,
        artifact_path="logistic_regression_iris",
        signature=llr_signature
    )

print("Run terminé ! Vérifiez l'interface à l'adresse http://15.236.42.9:5000")