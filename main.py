"""
main.py

Point d'entrée du pipeline de Machine Learning
pour la prédiction du churn client.

Usage:
    python main.py --data Churn_Modelling.csv
    python main.py --data Churn_Modelling.csv --prepare_data
    python main.py --data Churn_Modelling.csv --train_model
    python main.py --data Churn_Modelling.csv --evaluate_model
    python main.py --data Churn_Modelling.csv --save_model
"""

import argparse

from model_pipeline import (
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pipeline ML - Customer Churn Prediction"
    )

    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Chemin vers le fichier CSV"
    )

    parser.add_argument(
        "--save",
        type=str,
        default="classifier.joblib",
        help="Chemin de sauvegarde du modèle"
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion des données de test"
    )

    parser.add_argument(
        "--n-estimators",
        type=int,
        default=100,
        help="Nombre d'arbres Random Forest"
    )

    # Individual pipeline steps
    parser.add_argument(
        "--prepare_data",
        action="store_true",
        help="Execute data preparation"
    )

    parser.add_argument(
        "--train_model",
        action="store_true",
        help="Execute model training"
    )

    parser.add_argument(
        "--evaluate_model",
        action="store_true",
        help="Execute model evaluation"
    )

    parser.add_argument(
        "--save_model",
        action="store_true",
        help="Save the trained model"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # If no specific step is selected, execute everything
    run_all = not any([
        args.prepare_data,
        args.train_model,
        args.evaluate_model,
        args.save_model
    ])

    # STEP 1: Prepare data
    print("\n=== STEP 1: DATA PREPARATION ===")

    if run_all or args.prepare_data:
        x_train, x_test, y_train, y_test, scaler = prepare_data(
            args.data,
            test_size=args.test_size
        )
    else:
        # Training, evaluation and saving depend on preparation
        x_train, x_test, y_train, y_test, scaler = prepare_data(
            args.data,
            test_size=args.test_size
        )

    # STEP 2: Train model
    model = None

    if run_all or args.train_model or args.evaluate_model or args.save_model:
        print("\n=== STEP 2: MODEL TRAINING ===")

        model = train_model(
            x_train,
            y_train,
            n_estimators=args.n_estimators
        )

    # STEP 3: Evaluate model
    if run_all or args.evaluate_model:
        print("\n=== STEP 3: MODEL EVALUATION ===")

        evaluate_model(
            model,
            x_test,
            y_test
        )

    # STEP 4: Save model
    if run_all or args.save_model:
        print("\n=== STEP 4: MODEL SAVING ===")

        save_model(
            model,
            args.save,
            scaler
        )

    print("\nPipeline execution completed successfully!")


if __name__ == "__main__":
    main()
