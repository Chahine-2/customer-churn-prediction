#importing librries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn import metrics

import joblib


def prepare_data(filepath, test_size=0.2, random_state=1):
    df = pd.read_csv(filepath)

    # Détection et suppression des doublons
    nb_duplicates = df.duplicated().sum()
    if nb_duplicates > 0:
        print(f"{nb_duplicates} doublon(s) détecté(s) et supprimé(s).")
        df = df.drop_duplicates()
    else:
        print("Aucun doublon détecté.")

    # Détection et traitement des valeurs manquantes
    missing_percentages = df.isna().sum().sort_values(ascending=False) / len(df)
    nb_missing = df.isna().sum().sum()
    if nb_missing > 0:
        print(f"{nb_missing} valeur(s) manquante(s) détectée(s) :")
        print(missing_percentages[missing_percentages > 0])
        df = df.dropna()
    else:
        print("Aucune valeur manquante détectée.")

    # Encodage de la colonne Gender
    encoder = LabelEncoder()
    df["Gender"] = encoder.fit_transform(df["Gender"])

    # Suppression des colonnes non pertinentes pour le modèle
    columns_to_drop = ["Surname", "Geography"]
    df = df.drop(columns_to_drop, axis=1)

    # Séparation features / cible
    x = df.drop(["Exited"], axis=1)
    y = df["Exited"]
    x = x.drop(columns=["RowNumber", "CustomerId"])

    # Split train / test
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state
    )

    # Standardisation des données
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    return x_train_scaled, x_test_scaled, y_train, y_test, scaler


def train_model(x_train, y_train, n_estimators=100, random_state=42):
    model = RandomForestClassifier(
        n_estimators=n_estimators, random_state=random_state
    )
    model.fit(x_train, y_train)
    return model




def evaluate_model(model, x_test, y_test):
    y_pred = model.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)

    print(f"Accuracy score: {accuracy * 100:.2f}%")
    print("Confusion matrix:")
    print(matrix)

    return {"accuracy": accuracy, "confusion_matrix": matrix}


def save_model(model, filepath="classifier.joblib"):
    joblib.dump(model, filepath)
    print(f"Modèle sauvegardé dans : {filepath}")


def load_model(filepath="classifier.joblib"):
    model = joblib.load(filepath)
    print(f"Modèle chargé depuis : {filepath}")
    return model
