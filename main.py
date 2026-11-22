
"""
main.py

Entry point for the Customer Churn Prediction ML pipeline.

Usage:
    python main.py --data Churn_Modelling.csv

    python main.py --data Churn_Modelling.csv --prepare_data
    python main.py --data Churn_Modelling.csv --train_model
    python main.py --data Churn_Modelling.csv --evaluate_model
    python main.py --data Churn_Modelling.csv --save_model
    python main.py --data Churn_Modelling.csv --load_model
"""

import argparse

from model_pipeline import (
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
    load_model
)


def parse_args():

    parser = argparse.ArgumentParser(
        description="Machine Learning Pipeline - Customer Churn Prediction"
    )

    # Dataset
    parser.add_argument(
        "--data",
        type=str,
        default="Churn_Modelling.csv",
        help="Path to the dataset"
    )

    # Model file
    parser.add_argument(
        "--save",
        type=str,
        default="classifier.joblib",
        help="Path to save or load the model"
    )

    # Training parameters
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test dataset proportion"
    )

    parser.add_argument(
        "--n-estimators",
        type=int,
        default=100,
        help="Number of trees in Random Forest"
    )

    # Pipeline operations
    parser.add_argument(
        "--prepare_data",
        action="store_true",
        help="Prepare the dataset"
    )

    parser.add_argument(
        "--train_model",
        action="store_true",
        help="Train the model"
    )

    parser.add_argument(
        "--evaluate_model",
        action="store_true",
        help="Evaluate the model"
    )

    parser.add_argument(
        "--save_model",
        action="store_true",
        help="Save the trained model and scaler"
    )

    parser.add_argument(
        "--load_model",
        action="store_true",
        help="Load a previously saved model and scaler"
    )

    return parser.parse_args()


def main():

    args = parse_args()

    # Check selected operations
    selected_steps = [
        args.prepare_data,
        args.train_model,
        args.evaluate_model,
        args.save_model,
        args.load_model
    ]

    if sum(selected_steps) > 1:
        print("Error: Please select only one operation at a time.")
        return

    # LOAD MODEL
    if args.load_model:

        print("\n=== LOADING SAVED MODEL ===")

        model, scaler = load_model(args.save)

        print("\nModel and scaler loaded successfully!")

        return

    # Determine whether to execute the complete pipeline
    run_all = not any(selected_steps)

    # STEP 1: DATA PREPARATION
    print("\n=== STEP 1: DATA PREPARATION ===")

    x_train, x_test, y_train, y_test, scaler = prepare_data(
        args.data,
        test_size=args.test_size
    )

    if args.prepare_data and not run_all:
        print("\nData preparation completed successfully!")
        return

    # STEP 2: MODEL TRAINING
    print("\n=== STEP 2: MODEL TRAINING ===")

    model = train_model(
        x_train,
        y_train,
        n_estimators=args.n_estimators
    )

    if args.train_model and not run_all:
        print("\nModel training completed successfully!")
        return

    # STEP 3: MODEL EVALUATION
    if run_all or args.evaluate_model:

        print("\n=== STEP 3: MODEL EVALUATION ===")

        evaluate_model(
            model,
            x_test,
            y_test
        )

    if args.evaluate_model and not run_all:
        print("\nModel evaluation completed successfully!")
        return

    # STEP 4: MODEL SAVING
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
