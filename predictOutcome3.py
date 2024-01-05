import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, accuracy_score
from math import sqrt
from joblib import dump, load
from sklearn.metrics import r2_score


def preprocess_possession_column(data, column_name):
    def convert_possession(value):
        if isinstance(value, str) and '%' in value:
            return float(value.rstrip('%')) / 100.0
        else:
            return float(value) / 100

    data[column_name] = data[column_name].apply(convert_possession)
    return data


def train_decision_tree_model(X_train, y_train, model_filename='decision_tree_model.joblib'):
    dt_model = DecisionTreeRegressor()
    dt_model.fit(X_train, y_train)
    dump(dt_model, model_filename)
    return dt_model


def train_random_forest_model(X_train, y_train, model_filename='random_forest_model.joblib'):
    rf_model = RandomForestRegressor()
    rf_model.fit(X_train, y_train)
    dump(rf_model, model_filename)
    return rf_model


def train_linear_regression_model(X_train, y_train, model_filename='linear_regression_model.joblib'):
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    dump(lr_model, model_filename)
    return lr_model


def load_model(model_filename):
    return load(model_filename)

def make_predictions(models, X_test):
    predictions = []
    for model in models:
        predictions.append(model.predict(X_test))
    return predictions

def preprocess_data(data):
    try:
        data = data.drop(['team_home', 'team_away'], axis=1)
    except:
        pass
    data = preprocess_possession_column(data, 'possession_home')
    data = preprocess_possession_column(data, 'possession_away')
    return data

def predict(models, new_features):
    new_data = pd.DataFrame([new_features])
    new_data = preprocess_data(new_data)
    predictions = make_predictions(models, new_data)
    return predictions

def round_predictions(predictions):
    rounded_predictions = []
    for prediction in predictions:
        rounded_predictions.append(prediction.round().astype(int))
    return rounded_predictions

def format_score(prediction):
    if len(prediction) == 1:
        return f"home {prediction[0]} : away 0"
    else:
        return f"home {prediction[0]} : away {prediction[1]}"



def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    r2 = r2_score(y_test, predictions, multioutput='variance_weighted')
    return r2


def train_and_evaluate_model(X_train, y_train, X_test, y_test, model_filename, retrain_threshold=0.8, max_retries=500):
    model = train_decision_tree_model(X_train, y_train)  # You can choose any model here
    r2 = evaluate_model(model, X_test, y_test)

    if r2 > retrain_threshold:
        print(f"Model R-squared Score: {r2}. Saving the model.")
        dump(model, model_filename)
    else:
        print(f"Model R-squared Score: {r2}. Retraining the model.")
        if max_retries > 0:
            train_and_evaluate_model(X_train, y_train, X_test, y_test, model_filename, retrain_threshold,
                                     max_retries - 1)
        else:
            print("Maximum retries reached. Stopping retraining.")


if __name__ == "__main__":
    data = pd.read_excel('football_team_training_data.xlsx')
    data = preprocess_data(data)

    features = ['possession_home', 'possession_away', 'shots_on_target_home', 'shots_on_target_away']
    target = ['goals_home', 'goals_away']

    X = data[features]
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    dt_model = train_decision_tree_model(X_train, y_train)
    rf_model = train_random_forest_model(X_train, y_train)
    lr_model = train_linear_regression_model(X_train, y_train)

    loaded_dt_model = load_model('decision_tree_model.joblib')
    loaded_rf_model = load_model('random_forest_model.joblib')
    loaded_lr_model = load_model('linear_regression_model.joblib')

    models = [loaded_dt_model, loaded_rf_model, loaded_lr_model]
    # models = [ loaded_lr_model]

    for i, model in enumerate(models):
        r2 = evaluate_model(model, X_test, y_test)
        print(f"Model {i + 1} R-squared Score: {r2}")

        # Specify the R-squared threshold (e.g., 0.8)
        r2_threshold = 0.55

        if r2 > r2_threshold:
            print(f"Model {i + 1} R-squared score is above the threshold. Saving the model.")
            dump(model, f'model_{i + 1}.joblib')
        else:
            print(f"Model {i + 1} R-squared score is below the threshold. Retraining the model.")
            train_and_evaluate_model(X_train, y_train, X_test, y_test, f'model_{i + 1}.joblib', r2_threshold)

    new_features = {'possession_home': '45%', 'possession_away': 0.46, 'shots_on_target_home': 8, 'shots_on_target_away': 5}
    new_predictions = predict(models, new_features)
    rounded_predictions = round_predictions(new_predictions)

    for i, prediction in enumerate(rounded_predictions):
        print(f"Model {i + 1} Prediction: {format_score(prediction)}")

    final_prediction = format_score(sum(rounded_predictions) / len(rounded_predictions))
    print("Final Prediction:", final_prediction)
