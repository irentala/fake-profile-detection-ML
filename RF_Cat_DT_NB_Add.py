import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np

# Load the dataset
file_path = '/Users/irentala/PycharmProjects/fake-profile-detection-ML/data/cleansed_data_33.csv'
df = pd.read_csv(file_path, sep=',')

# Feature extraction: Creating new features for modeling
# Compute Hold Time as (release_time - press_time)
df['hold_time'] = df['release_time'] - df['press_time']

# Compute Flight Time (time between consecutive key presses)
df['flight_time'] = df['press_time'].diff().fillna(0)
df.loc[df['session_id'] != df['session_id'].shift(), 'flight_time'] = 0  # Reset flight time at session boundaries

# Compute Preceding Flight Time (time between release of previous key and press of next key)
df['preceding_flight_time'] = df['press_time'] - df['release_time'].shift().fillna(0)
df.loc[df['session_id'] != df['session_id'].shift(), 'preceding_flight_time'] = 0  # Reset preceding flight time at session boundaries

# Compute Following Flight Time (time between release of current key and press of next key)
df['following_flight_time'] = df['press_time'].shift(-1) - df['release_time']
df['following_flight_time'] = df['following_flight_time'].fillna(0)
df.loc[df['session_id'] != df['session_id'].shift(-1), 'following_flight_time'] = 0  # Reset following flight time at session boundaries

# Compute Digraph Time (time taken to type two consecutive keys)
df['digraph_time'] = df['release_time'].shift(-1) - df['press_time']
df['digraph_time'] = df['digraph_time'].fillna(0)
df.loc[df['session_id'] != df['session_id'].shift(-1), 'digraph_time'] = 0

# Compute mean and standard deviation of hold time per session
df['hold_time_mean'] = df.groupby('session_id')['hold_time'].transform('mean')
df['hold_time_std'] = df.groupby('session_id')['hold_time'].transform('std').fillna(0)

# Compute mean and standard deviation of flight time per session
df['flight_time_mean'] = df.groupby('session_id')['flight_time'].transform('mean')
df['flight_time_std'] = df.groupby('session_id')['flight_time'].transform('std').fillna(0)

# Feature selection: Adding all computed features to the feature set
features = ['hold_time', 'flight_time', 'preceding_flight_time', 'following_flight_time', 'digraph_time',
            'hold_time_mean', 'hold_time_std', 'flight_time_mean', 'flight_time_std']
x = df[features]
y = df['user_ids']

# Apply StandardScaler
standard_scaler = StandardScaler()
x_standard = standard_scaler.fit_transform(x)

# Apply MinMaxScaler
minmax_scaler = MinMaxScaler()
x_minmax = minmax_scaler.fit_transform(x)

# Apply Extended MinMaxScaler (scaling to a custom range)
extended_minmax_scaler = MinMaxScaler(feature_range=(-1, 1))
x_extended_minmax = extended_minmax_scaler.fit_transform(x)


# Define a function to train and evaluate classifiers with different scalers
def evaluate_model(x_scaled, scaler_name):
    print(f"\nResults for {scaler_name}:\n")

    # Split the data into training and test sets (changing split ratio to 70-30)
    x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)

    # Random Forest Classifier
    rf_classifier = RandomForestClassifier(n_estimators=50, max_depth=10, min_samples_split=5, min_samples_leaf=3,
                                           random_state=42)
    rf_classifier.fit(x_train, y_train)
    y_pred_rf = rf_classifier.predict(x_test)
    rf_accuracy = accuracy_score(y_test, y_pred_rf)
    print("Random Forest Classifier Accuracy:", rf_accuracy)
    rf_report = classification_report(y_test, y_pred_rf, zero_division=1, output_dict=True)
    print("Random Forest Classification Report:\n", classification_report(y_test, y_pred_rf, zero_division=1))

    # Decision Tree Classifier
    dt_classifier = DecisionTreeClassifier(max_depth=10, min_samples_split=5, min_samples_leaf=3, random_state=42)
    dt_classifier.fit(x_train, y_train)
    y_pred_dt = dt_classifier.predict(x_test)
    dt_accuracy = accuracy_score(y_test, y_pred_dt)
    print("Decision Tree Classifier Accuracy:", dt_accuracy)
    dt_report = classification_report(y_test, y_pred_dt, zero_division=1, output_dict=True)
    print("Decision Tree Classification Report:\n", classification_report(y_test, y_pred_dt, zero_division=1))

    # CatBoost Classifier
    catboost_classifier = CatBoostClassifier(iterations=50, learning_rate=0.05, depth=4, l2_leaf_reg=3, verbose=0)
    catboost_classifier.fit(x_train, y_train)
    y_pred_catboost = catboost_classifier.predict(x_test)
    catboost_accuracy = accuracy_score(y_test, y_pred_catboost)
    print("CatBoost Classifier Accuracy:", catboost_accuracy)
    catboost_report = classification_report(y_test, y_pred_catboost, zero_division=1, output_dict=True)
    print("CatBoost Classification Report:\n", classification_report(y_test, y_pred_catboost, zero_division=1))

    # Naive Bayes Classifier
    nb_classifier = GaussianNB()
    nb_classifier.fit(x_train, y_train)
    y_pred_nb = nb_classifier.predict(x_test)
    nb_accuracy = accuracy_score(y_test, y_pred_nb)
    print("Naive Bayes Classifier Accuracy:", nb_accuracy)
    nb_report = classification_report(y_test, y_pred_nb, zero_division=1, output_dict=True)
    print("Naive Bayes Classification Report:\n", classification_report(y_test, y_pred_nb, zero_division=1))


    return (rf_accuracy, rf_report['weighted avg']), (dt_accuracy, dt_report['weighted avg']), \
           (catboost_accuracy, catboost_report['weighted avg']), (nb_accuracy, nb_report['weighted avg'])


# Evaluate models for each scaler
weighted_avgs = []
weighted_avgs.append(evaluate_model(x_standard, "StandardScaler"))
weighted_avgs.append(evaluate_model(x_minmax, "MinMaxScaler"))
weighted_avgs.append(evaluate_model(x_extended_minmax, "Extended MinMaxScaler"))

# Prepare results for Excel format
excel_data = {
    "Scaler": ["StandardScaler", "MinMaxScaler", "Extended MinMaxScaler"],
    "Random Forest": [
        f"Accuracy: {weighted_avgs[0][0][0]:.2f}, Precision: {weighted_avgs[0][0][1]['precision']:.2f}, Recall: {weighted_avgs[0][0][1]['recall']:.2f}, F1-score: {weighted_avgs[0][0][1]['f1-score']:.2f}",
        f"Accuracy: {weighted_avgs[1][0][0]:.2f}, Precision: {weighted_avgs[1][0][1]['precision']:.2f}, Recall: {weighted_avgs[1][0][1]['recall']:.2f}, F1-score: {weighted_avgs[1][0][1]['f1-score']:.2f}",
        f"Accuracy: {weighted_avgs[2][0][0]:.2f}, Precision: {weighted_avgs[2][0][1]['precision']:.2f}, Recall: {weighted_avgs[2][0][1]['recall']:.2f}, F1-score: {weighted_avgs[2][0][1]['f1-score']:.2f}"
    ],
    "Decision Tree": [
        f"Accuracy: {weighted_avgs[0][1][0]:.2f}, Precision: {weighted_avgs[0][1][1]['precision']:.2f}, Recall: {weighted_avgs[0][1][1]['recall']:.2f}, F1-score: {weighted_avgs[0][1][1]['f1-score']:.2f}",
        f"Accuracy: {weighted_avgs[1][1][0]:.2f}, Precision: {weighted_avgs[1][1][1]['precision']:.2f}, Recall: {weighted_avgs[1][1][1]['recall']:.2f}, F1-score: {weighted_avgs[1][1][1]['f1-score']:.2f}",
        f"Accuracy: {weighted_avgs[2][1][0]:.2f}, Precision: {weighted_avgs[2][1][1]['precision']:.2f}, Recall: {weighted_avgs[2][1][1]['recall']:.2f}, F1-score: {weighted_avgs[2][1][1]['f1-score']:.2f}"
    ],
    "CatBoost": [
        f"Accuracy: {weighted_avgs[0][2][0]:.2f}, Precision: {weighted_avgs[0][2][1]['precision']:.2f}, Recall: {weighted_avgs[0][2][1]['recall']:.2f}, F1-score: {weighted_avgs[0][2][1]['f1-score']:.2f}",
        f"Accuracy: {weighted_avgs[1][2][0]:.2f}, Precision: {weighted_avgs[1][2][1]['precision']:.2f}, Recall: {weighted_avgs[1][2][1]['recall']:.2f}, F1-score: {weighted_avgs[1][2][1]['f1-score']:.2f}",
        f"Accuracy: {weighted_avgs[2][2][0]:.2f}, Precision: {weighted_avgs[2][2][1]['precision']:.2f}, Recall: {weighted_avgs[2][2][1]['recall']:.2f}, F1-score: {weighted_avgs[2][2][1]['f1-score']:.2f}"
    ],
    "Naive Bayes": [
        f"Accuracy: {weighted_avgs[0][3][0]:.2f}, Precision: {weighted_avgs[0][3][1]['precision']:.2f}, Recall: {weighted_avgs[0][3][1]['recall']:.2f}, F1-score: {weighted_avgs[0][3][1]['f1-score']:.2f}",
        f"Accuracy: {weighted_avgs[1][3][0]:.2f}, Precision: {weighted_avgs[1][3][1]['precision']:.2f}, Recall: {weighted_avgs[1][3][1]['recall']:.2f}, F1-score: {weighted_avgs[1][3][1]['f1-score']:.2f}",
        f"Accuracy: {weighted_avgs[2][3][0]:.2f}, Precision: {weighted_avgs[2][3][1]['precision']:.2f}, Recall: {weighted_avgs[2][3][1]['recall']:.2f}, F1-score: {weighted_avgs[2][3][1]['f1-score']:.2f}"
    ]
}

# Create DataFrame and print in Excel format
excel_df = pd.DataFrame(excel_data)
print("\nResults in Excel Format:\n")
print(excel_df.to_string(index=False))
