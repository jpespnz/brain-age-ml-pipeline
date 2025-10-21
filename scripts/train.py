import argparse
import os
import joblib

from data_loader import load_data, split_data, preprocess_data
from baseline_model import train_baseline, evaluate_model


def main():
    """
    Command-line interface for training a baseline brain-age model.
    """
    parser = argparse.ArgumentParser(description="Train baseline brain-age model")
    parser.add_argument('--data', type=str, required=True, help='Path to CSV data file')
    parser.add_argument('--test_size', type=float, default=0.2, help='Fraction of data to use as test set')
    parser.add_argument('--model_path', type=str, default='models/baseline_model.joblib', help='Path to save the trained model')
    args = parser.parse_args()

    # Load and preprocess data
    X, y = load_data(args.data)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=args.test_size)
    X_train_scaled, X_test_scaled, scaler = preprocess_data(X_train, X_test)

    # Train baseline model
    model = train_baseline(X_train_scaled, y_train)

    # Evaluate model
    mae = evaluate_model(model, X_test_scaled, y_test)
    print(f"Mean Absolute Error: {mae:.2f}")

    # Ensure the directory for the model exists
    model_dir = os.path.dirname(args.model_path)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)

    # Save model and scaler together
    joblib.dump({'model': model, 'scaler': scaler}, args.model_path)
    print(f"Model saved to {args.model_path}")


if __name__ == '__main__':
    main()
