
import pandas as pd
import os

def load_data(filename='telecom_churn.csv'):
    """Charge le dataset depuis le dossier dataset"""
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(BASE_DIR, 'dataset', filename)
    df = pd.read_csv(DATA_PATH)
    return df

def preprocess_data(df):
    """Prépare X et y pour les modèles"""
    # Si les variables catégorielles existent, encoder
    df = pd.get_dummies(df, drop_first=True)
    
    # Vérifier le nom de la colonne target
    if 'Churn_Yes' in df.columns:
        X = df.drop('Churn_Yes', axis=1)
        y = df['Churn_Yes']
    else:
        X = df.drop('Churn', axis=1)
        y = df['Churn']
    return X, y

if __name__ == "__main__":
    df = load_data()
    print("Aperçu des données :")
    print(df.head())
    X, y = preprocess_data(df)
    print("\nShape de X:", X.shape)
    print("Shape de y:", y.shape)