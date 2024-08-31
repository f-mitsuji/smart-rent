import joblib
import optuna
import pandas as pd
from data_preprocessor import PROCESSED_DATA_PATH
from ml_settings import MODEL_DIR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

MODEL_PATH = MODEL_DIR / "suumo_rent_prediction_model.pkl"


def load_and_preprocess_data(filepath):
    property_df = pd.read_csv(filepath)
    numeric_features = [
        "floor",
        "management_fee",
        "deposit",
        "key_money",
        "area",
        "building_age",
        "total_floors",
        "total_transit_time",
    ]

    scaler = StandardScaler()
    property_df[numeric_features] = scaler.fit_transform(property_df[numeric_features])

    return property_df


def objective(trial):
    property_df = load_and_preprocess_data(PROCESSED_DATA_PATH)
    X = property_df.drop(columns=["rent"])
    y = property_df["rent"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    n_estimators = trial.suggest_int("n_estimators", 50, 500)
    max_depth = trial.suggest_int("max_depth", 5, 50)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)
    max_features = trial.suggest_categorical("max_features", ["sqrt", "log2", None])

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=1,
        n_jobs=-1,
    )

    scores = cross_val_score(model, X_train, y_train, cv=5, scoring="neg_mean_squared_error")
    mean_score = scores.mean()

    return -mean_score


def train_and_evaluate_model(property_df):
    X = property_df.drop(columns=["rent"])
    y = property_df["rent"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=100, timeout=600)

    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    best_model = RandomForestRegressor(**trial.params, random_state=1, n_jobs=-1)
    best_model.fit(X_train, y_train)

    y_pred = best_model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    print(f"R^2 Score: {r2}")
    print(f"Mean Squared Error: {mse}")

    cv_scores_r2 = cross_val_score(best_model, X_train, y_train, cv=10, scoring="r2")
    cv_scores_mse = cross_val_score(best_model, X_train, y_train, cv=10, scoring="neg_mean_squared_error")

    print(f"Average Cross-Validation R^2 Score: {cv_scores_r2.mean():.4f} (+/- {cv_scores_r2.std() * 2:.4f})")
    print(f"Average Cross-Validation MSE Score: {-cv_scores_mse.mean():.4f} (+/- {cv_scores_mse.std() * 2:.4f})")

    return best_model


def save_model(model, model_filename):
    joblib.dump(model, model_filename)
    print(f"Model saved to {model_filename}")


def main():
    print("Loading and preprocessing data...")
    property_df = load_and_preprocess_data(PROCESSED_DATA_PATH)

    print("\nOptimizing hyperparameters and training model...")
    model = train_and_evaluate_model(property_df)

    print("\nSaving model...")
    save_model(model, MODEL_PATH)


if __name__ == "__main__":
    main()
