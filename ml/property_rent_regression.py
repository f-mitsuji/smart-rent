import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler


def load_and_preprocess_data(filepath):
    property_df = pd.read_csv(filepath)

    median_floor = property_df["Room Floor"].median()
    property_df["Room Floor"] = property_df["Room Floor"].fillna(median_floor)

    numeric_features = [
        "Room Floor",
        "Management Fee",
        "Deposit",
        "Gratuity",
        "Area",
        "Building Age",
        "Total Floors",
        "Total Transit Time",
    ]
    scaler = StandardScaler()
    property_df[numeric_features] = scaler.fit_transform(property_df[numeric_features])

    return property_df


def train_and_evaluate_model(df):
    X = df.drop(columns=["Rent"])
    y = df["Rent"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error: {mse}")
    print(f"R^2 Score: {r2}")

    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="r2")
    print(f"Cross-Validation R^2 Scores: {cv_scores}")
    print(f"Average Cross-Validation R^2 Score: {cv_scores.mean()}")

    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="neg_mean_squared_error")
    print(f"Cross-Validation MSE Scores: {-cv_scores}")
    print(f"Average Cross-Validation MSE Score: {-cv_scores.mean()}")

    return model, X_train.columns


def analyze_feature_importance(model, feature_names):
    importances = model.feature_importances_
    importance_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    importance_df = importance_df.sort_values(by="Importance", ascending=False)

    print("\nFeature Importances:")
    print(importance_df)

    return importance_df


def save_model(model, model_filename):
    joblib.dump(model, model_filename)
    print(f"Model saved to {model_filename}")


def main():
    filepath = "ml/data/processed/property_data_cleaned.csv"
    property_df = load_and_preprocess_data(filepath)

    model, feature_names = train_and_evaluate_model(property_df)
    analyze_feature_importance(model, feature_names)

    save_model(model, "ml/model/rent_prediction_model.pkl")


if __name__ == "__main__":
    main()
