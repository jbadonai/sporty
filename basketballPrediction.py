import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib

class PredictOutcome:
    def __init__(self):
        self.model = None

    def load_data(self, file_path):
        # Load data from the Excel file
        data = pd.read_excel(file_path)
        return data

    def preprocess_data(self, data):
        # Drop unnecessary columns
        data = data.drop(['Team name', 'Opponent name'], axis=1)

        def convert_possession(value):
            if isinstance(value, str) and '%' in value:
                return float(value.rstrip('%')) / 100.0
            else:
                return float(value) / 100

        # Convert percentage columns to numeric
        percentage_columns = ['Team FG%', 'Team FT%', 'Team 3P%', 'Opponent FG%', 'Opponent FT%', 'Opponent 3P%']
        for col in percentage_columns:
            data[col] = data[col].apply(convert_possession)

        # Split the 'Team quarterly scores' and 'Opponent quarterly scores' into separate columns
        quarterly_columns = ['Q1', 'Q2', 'Q3', 'Q4', 'Opponent Q1', 'Opponent Q2', 'Opponent Q3', 'Opponent Q4']
        for col in quarterly_columns:
            data[col] = data['Team quarterly scores' if col.startswith('Q') else 'Opponent quarterly scores'].apply(
                lambda x: sum(map(int, x.split('-'))) if isinstance(x, str) else x
            )

        # Drop the original quarterly scores columns
        data = data.drop(['Team quarterly scores', 'Opponent quarterly scores'], axis=1)

        return data

    def train_model_old(self, data):
        # Separate features and targets
        X = data.drop(
            ['Team total points', 'Opponent total points', 'Q1', 'Q2', 'Q3', 'Q4', 'Opponent Q1', 'Opponent Q2',
             'Opponent Q3', 'Opponent Q4'], axis=1)

        # Targets
        y_team_points = data['Team total points']
        y_opponent_points = data['Opponent total points']

        y_team_quarters = data[['Q1', 'Q2', 'Q3', 'Q4']]
        y_opponent_quarters = data[['Opponent Q1', 'Opponent Q2', 'Opponent Q3', 'Opponent Q4']]

        # Split the data into training and testing sets
        X_train, X_test, y_train_team_points, y_test_team_points, y_train_opponent_points, y_test_opponent_points, y_train_team_quarters, y_test_team_quarters, y_train_opponent_quarters, y_test_opponent_quarters = train_test_split(
            X, y_team_points, y_opponent_points, y_team_quarters, y_opponent_quarters, test_size=0.2, random_state=42
        )

        # Train models using different algorithms
        models = [
            RandomForestRegressor(),
            LinearRegression()
            # Add more models as needed
        ]

        best_model = None
        best_mse = float('inf')

        for model in models:
            # Train on total points
            model.fit(X_train, y_train_team_points)
            y_pred_team_points = model.predict(X_test)
            mse_team_points = mean_squared_error(y_test_team_points, y_pred_team_points)

            model.fit(X_train, y_train_opponent_points)
            y_pred_opponent_points = model.predict(X_test)
            mse_opponent_points = mean_squared_error(y_test_opponent_points, y_pred_opponent_points)

            # Train on quarters
            model.fit(X_train, y_train_team_quarters)
            y_pred_team_quarters = model.predict(X_test)
            mse_team_quarters = mean_squared_error(y_test_team_quarters, y_pred_team_quarters)

            model.fit(X_train, y_train_opponent_quarters)
            y_pred_opponent_quarters = model.predict(X_test)
            mse_opponent_quarters = mean_squared_error(y_test_opponent_quarters, y_pred_opponent_quarters)

            # Calculate average MSE
            average_mse = (mse_team_points + mse_opponent_points + mse_team_quarters + mse_opponent_quarters) / 4

            if average_mse < best_mse:
                best_mse = average_mse
                best_model = model

        self.model = best_model
        return best_model

    def check_accuracy(self, y_true, y_pred, model_name):
        r_squared = r2_score(y_true, y_pred)
        accuracy_percentage = r_squared * 100
        print(f"{model_name} R-squared score: {accuracy_percentage:.2f}%")

    def train_model(self, data):
        # Separate features and targets
        X = data.drop(
            ['Team total points', 'Opponent total points', 'Q1', 'Q2', 'Q3', 'Q4', 'Opponent Q1', 'Opponent Q2',
             'Opponent Q3', 'Opponent Q4'], axis=1)

        # Targets
        targets = {
            'team_points': 'Team total points',
            'opponent_points': 'Opponent total points',
            'team_quarters': ['Q1', 'Q2', 'Q3', 'Q4'],
            'opponent_quarters': ['Opponent Q1', 'Opponent Q2', 'Opponent Q3', 'Opponent Q4']
        }

        models = {}

        for target_name, target_columns in targets.items():
            y = data[target_columns] if isinstance(target_columns, list) else data[target_columns]

            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Train models using different algorithms
            model = RandomForestRegressor()
            # model = RandomForestRegressor() if 'team' in target_name.lower() else LinearRegression()

            # Train the model
            model.fit(X_train, y_train)

            # Save the trained model to disk
            model_filename = f"{target_name}_model.joblib"
            joblib.dump(model, model_filename)

            # Store the model in the dictionary
            models[target_name] = model_filename

        self.models = models
        return models

    def save_model(self, model, file_path):
        # Save the trained model to disk
        joblib.dump(model, file_path)

    def load_model(self, file_path):
        # Load the trained model from disk
        self.model = joblib.load(file_path)

    def predict(self, input_data):
        # Drop 'Team' and 'Opponent' columns
        input_data = input_data.drop(['Team', 'Opponent'], axis=1)

        input_all_features = input_data[
            ['Team FG%', 'Team FT%', 'Team 3P%', 'Opponent FG%', 'Opponent FT%', 'Opponent 3P%']]

        predictions = {}

        for target_name, model_filename in self.models.items():
            # Load the trained model from disk
            model = joblib.load(model_filename)

            # Make predictions using all features
            predictions[target_name] = model.predict(input_all_features)

            # Check if predictions is a 2D array
            if predictions[target_name].ndim == 2:
                # Take the first row if the predictions are in the shape (1, 4)
                predictions[target_name] = predictions[target_name][0]


            # Reset the index of input_data before assignment
            input_data = input_data.reset_index(drop=True)

            # Assign predictions directly to target_name
            if len(predictions[target_name]) > 1:
                result_string = ' - '.join(map(str, predictions[target_name]))
            else:
                result_string = str(predictions[target_name][0])

            print(target_name, end=":")
            print(result_string)

            input_data[target_name] = result_string

        return input_data




# Example usage:
# Initialize the PredictOutcome class
predictor = PredictOutcome()

# Load and preprocess the data
data = predictor.load_data("basketball_training_data.xlsx")
processed_data = predictor.preprocess_data(data)

# Train the model and save it to disk
print("Training models....")
trained_model = predictor.train_model(processed_data)
predictor.save_model(trained_model, "trained_model.joblib")

# Load the trained model from disk
print("Loading trained model...")
predictor.load_model("trained_model.joblib")

# Prepare input data for prediction
input_data = {
    'team_home': ['Kinlung Pegasus'],
    'team_away': ['South China'],
    'field_goal_percentage_home': [34.1],
    'free_throw_percentage_home': [70.0],
    'three_point_percentage_home': [26.8],
    'field_goal_percentage_away': [48.4],
    'free_throw_percentage_away': [78.5],
    'three_point_percentage_away': [35.1]
}

# Mapping dictionary for reordering keys
feature_mapping = {
    'field_goal_percentage_home': 'Team FG%',
    'free_throw_percentage_home': 'Team FT%',
    'three_point_percentage_home': 'Team 3P%',
    'field_goal_percentage_away': 'Opponent FG%',
    'free_throw_percentage_away': 'Opponent FT%',
    'three_point_percentage_away': 'Opponent 3P%',
}

# Reorder and rename keys in input_data
reordered_input_data = {
    feature_mapping[key]: value for key, value in input_data.items() if key in feature_mapping
}


# Add team and opponent names
reordered_input_data['Team'] = input_data['team_home']
reordered_input_data['Opponent'] = input_data['team_away']

# Create a DataFrame from reordered input data
input_df = pd.DataFrame(reordered_input_data)

# print(input_df)
# input("wait...")

# Make predictions
predictions = predictor.predict(input_df)

# Display the predictions
# print(predictions.columns)
# print(predictions)

