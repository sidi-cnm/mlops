import pandas as pd
import yaml
import os

## 1. Chargement des paramètres (Seulement preprocess)
params = yaml.safe_load(open("params.yaml"))['preprocess']

def save_df(df, path):
    """
    Sauvegarde le DataFrame. 
    Si le chemin commence par s3://, pandas utilisera les credentials 
    système automatiquement.
    """
    print(f"Sauvegarde en cours vers : {path}...")
    # Pas besoin de storage_options si tes clés sont configurées dans le système
    df.to_csv(path, index=False)

def preprocess(input_path, output_path):
    # Créer le dossier local si nécessaire (pour éviter les erreurs dvc)
    if not output_path.startswith("s3://"):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Lecture des données
    data = pd.read_csv(input_path)
    
    # Preprocessing
    data = data.dropna()
    
    # Sauvegarde
    save_df(data, output_path)
    print(f"Preprocessed data saved to: {output_path}")

if __name__ == "__main__":
    # On ne passe plus aws_params
    preprocess(params["input"], params["output"])