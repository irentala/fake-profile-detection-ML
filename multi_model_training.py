import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s]: %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)


# Function to load and preprocess the dataset
def load_and_preprocess_data(file_path, scaler_type):
    logging.info(f"Loading and preprocessing the dataset with {scaler_type} scaler...")
    # Load dataset
    df = pd.read_csv(file_path, sep=',')

    # Feature extraction
    df['hold_time'] = df['release_time'] - df['press_time']
    df['flight_time'] = df['press_time'].diff().fillna(0)
    df.loc[df['session_id'] != df['session_id'].shift(), 'flight_time'] = 0
    df['preceding_flight_time'] = df['press_time'] - df['release_time'].shift().fillna(0)
    df.loc[df['session_id'] != df['session_id'].shift(), 'preceding_flight_time'] = 0
    df['following_flight_time'] = df['press_time'].shift(-1) - df['release_time']
    df['following_flight_time'] = df['following_flight_time'].fillna(0)
    df.loc[df['session_id'] != df['session_id'].shift(-1), 'following_flight_time'] = 0

    # Select features and target
    features = ['hold_time', 'flight_time', 'preceding_flight_time', 'following_flight_time']
    x = df[features]
    y = df['user_ids']

    # Apply the specified scaler
    if scaler_type == "standard":
        scaler = StandardScaler()
    elif scaler_type == "minmax":
        scaler = MinMaxScaler()
    elif scaler_type == "extended_minmax":
        scaler = MinMaxScaler(feature_range=(-1, 1))
    else:
        raise ValueError(f"Unsupported scaler type: {scaler_type}")

    x_scaled = scaler.fit_transform(x)
    logging.info(f"Preprocessing with {scaler_type} scaler completed.")
    return x_scaled, y


# Function to train and evaluate a model
def evaluate_model(x_train, x_test, y_train, y_test, model, model_name, scaler_name):
    logging.info(f"Training {model_name} with {scaler_name} scaler...")
    model.fit(x_train, y_train)
    logging.info(f"Training completed for {model_name} with {scaler_name} scaler.")
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    logging.info(f"{model_name} with {scaler_name} scaler - Accuracy: {accuracy:.2f}")
    logging.info(f"Generating classification report for {model_name} with {scaler_name} scaler...")
    print(f"\n{model_name} Classification Report (Scaler: {scaler_name}):\n",
          classification_report(y_test, y_pred, zero_division=1))


# Main function to handle multiple models and scalers
def main(models, scalers):
    # File path to the dataset
    file_path = '/Users/irentala/PycharmProjects/fake-profile-detection-transformer/data/cleansed_data_33.csv'

    for scaler in scalers:
        # Load and preprocess data for the given scaler
        x, y = load_and_preprocess_data(file_path, scaler)

        # Split data into training and testing sets
        logging.info("Splitting data into training and testing sets...")
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        for model_name in models:
            if model_name == 'random_forest':
                model = RandomForestClassifier(
                    n_estimators=50, max_depth=10, min_samples_split=5, min_samples_leaf=3, random_state=42
                )
            elif model_name == 'catboost':
                model = CatBoostClassifier(
                    iterations=50, learning_rate=0.05, depth=4, l2_leaf_reg=3, verbose=0
                )
            elif model_name == 'decision_tree':
                model = DecisionTreeClassifier(
                    max_depth=10, min_samples_split=5, random_state=42
                )
            elif model_name == 'naive_bayes':
                model = GaussianNB()
            elif model_name == 'svm':
                model = SVC(kernel='rbf', random_state=42)
            else:
                logging.error(f"Unknown model: {model_name}. Skipping.")
                continue

            # Train and evaluate the model
            evaluate_model(x_train, x_test, y_train, y_test, model, model_name, scaler)


# Entry point for the script
if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run multiple classifiers with multiple scalers.")
    parser.add_argument(
        "--models",
        nargs='+',
        default=["random_forest"],
        help="Specify the models to run (e.g., random_forest catboost decision_tree). Default is 'random_forest'."
    )
    parser.add_argument(
        "--scalers",
        nargs='+',
        default=["standard"],
        help="Specify the scalers to use (e.g., standard minmax extended_minmax). Default is 'standard'."
    )
    args = parser.parse_args()

    # Run the main function with the specified models and scalers
    main(args.models, args.scalers)
