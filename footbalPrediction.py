# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error
# import joblib
#
# class PredictOutcome:
#     def __init__(self, algorithm='random_forest'):
#         self.algorithm = algorithm
#         self.model = None
#
#     def _process_possession(self, possession):
#         # Convert possession to percentage
#         if isinstance(possession, str) and '%' in possession:
#             return float(possession.rstrip('%')) / 100.0
#         else:
#             return float(possession) / 100
#
#     def _prepare_data(self, data):
#         # Convert possession columns to numeric values
#         data['possession_home'] = data['possession_home'].apply(self._process_possession)
#         data['possession_away'] = data['possession_away'].apply(self._process_possession)
#         return data
#
#     def train_model(self, file_path='football_team_training_data.xlsx'):
#         # Load data from Excel file
#         training_data = pd.read_excel(file_path)
#
#         # Prepare data for training
#         training_data = self._prepare_data(training_data)
#
#         # Split data into features and target
#         features = training_data[['possession_home', 'possession_away', 'shots_on_target_home', 'shots_on_target_away']]
#         target = training_data[['goals_home', 'goals_away']]
#
#         # Split data into training and testing sets
#         X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
#
#         # Choose the algorithm
#         if self.algorithm == 'random_forest':
#             model = RandomForestRegressor()
#         else:
#             model = LinearRegression()
#             # Add more conditions for other algorithms if needed
#             pass
#
#         # Train the model
#         model.fit(X_train, y_train)
#
#         # Evaluate the model
#         predictions = model.predict(X_test)
#         mse = mean_squared_error(y_test, predictions)
#         print(f'Mean Squared Error: {mse}')
#
#         # Save the trained model to disk
#         joblib.dump(model, 'trained_model.pkl')
#
#     def load_model(self, model_path='trained_model.pkl'):
#         # Load the pre-trained model from disk
#         self.model = joblib.load(model_path)
#
#     def predict_scores(self, input_data):
#         # Ensure the model is loaded
#         if self.model is None:
#             print("Error: Model not loaded. Please load the model before making predictions.")
#             return None
#
#         # Prepare input data
#         input_data['possession_home'] = self._process_possession(input_data['possession_home'])
#         input_data['possession_away'] = self._process_possession(input_data['possession_away'])
#
#         # Make prediction
#         prediction = self.model.predict([[
#             input_data['possession_home'],
#             input_data['possession_away'],
#             input_data['shots_on_target_home'],
#             input_data['shots_on_target_away']
#         ]])
#
#         return {
#             'predicted_goals_home': prediction[0][0],
#             'predicted_goals_away': prediction[0][1]
#         }
#
# # Example usage:
# if __name__ == "__main__":
#     # Instantiate the PredictOutcome class
#     predictor = PredictOutcome()
#
#     # Train the model
#     predictor.train_model()
#
#     # Load the trained model
#     predictor.load_model()
#
#     # Sample data for prediction
#     input_data = {
#         'team_home': 'Tokyo Verdy Beleza',
#         'team_away': 'INAC Kobe Leonessa',
#         'possession_home': 53.67,
#         'possession_away': 53.33,
#         'shots_on_target_home': 6.33,
#         'shots_on_target_away': 6
#     }
#
#     # Make prediction
#     result = predictor.predict_scores(input_data)
#     print(result)


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib

class PredictOutcome:
    def __init__(self):
        self.model = None
        self.algorithm = None

    def _process_possession(self, possession):
        # Convert possession to percentage
        if isinstance(possession, str) and '%' in possession:
            v = float(possession.rstrip('%'))
            if v < 1:
                return v
            else:
                return v / 100.0
        else:
            v = float(possession)
            if v < 1:
                return v
            else:
                return v / 100

    def _prepare_data(self, data):
        # Convert possession columns to numeric values
        data['possession_home'] = data['possession_home'].apply(self._process_possession)
        data['possession_away'] = data['possession_away'].apply(self._process_possession)
        return data

    def _train_model(self, model, features, target):
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

        # Train the model
        model.fit(X_train, y_train)

        # Evaluate the model
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        return mse

    def train_model(self, file_path='football_team_training_data.xlsx'):
        # Load data from Excel file
        training_data = pd.read_excel(file_path)

        # Prepare data for training
        training_data = self._prepare_data(training_data)

        # Split data into features and target
        features = training_data[['possession_home', 'possession_away', 'shots_on_target_home', 'shots_on_target_away']]
        target = training_data[['goals_home', 'goals_away']]

        # Train Linear Regression model
        linear_regression = LinearRegression()
        linear_regression_mse = self._train_model(linear_regression, features, target)
        print(f'Linear Regression Mean Squared Error: {linear_regression_mse}')

        # Train Random Forest model
        random_forest = RandomForestRegressor(random_state=42)
        random_forest_mse = self._train_model(random_forest, features, target)
        print(f'Random Forest Mean Squared Error: {random_forest_mse}')

        # Choose the algorithm with the lower MSE
        if linear_regression_mse < random_forest_mse:
            self.model = linear_regression
            self.algorithm = 'linear_regression'
        else:
            self.model = random_forest
            self.algorithm = 'random_forest'

        # Save the trained model to disk
        joblib.dump(self.model, 'trained_model.pkl')
        print(f'Training completed. Selected algorithm: {self.algorithm}')

    def load_model(self, model_path='trained_model.pkl'):
        # Load the pre-trained model from disk
        self.model = joblib.load(model_path)
        print(f'Model loaded. Algorithm: {self.algorithm}')

    def predict_scores(self, input_data):
        # Ensure the model is loaded
        if self.model is None:
            print("Error: Model not loaded. Please load the model before making predictions.")
            return None

        if type(input_data['possession_home']) is list:
            # Prepare input data
            input_data = {
                'team_home': input_data['team_home'][0],
                'team_away': input_data['team_away'][0],
                'possession_home': input_data['possession_home'][0],
                'possession_away': input_data['possession_away'][0],
                'shots_on_target_home': input_data['shots_on_target_home'][0],
                'shots_on_target_away': input_data['shots_on_target_away'][0]
            }

        # Prepare input data
        input_data['possession_home'] = self._process_possession(input_data['possession_home'])
        input_data['possession_away'] = self._process_possession(input_data['possession_away'])

        # Make prediction
        prediction = self.model.predict([[
            input_data['possession_home'],
            input_data['possession_away'],
            input_data['shots_on_target_home'],
            input_data['shots_on_target_away']
        ]])

        return {
            'home_team_name': input_data['team_home'],
            'away_team_name': input_data['team_away'],
            'predicted_goals_home': prediction[0][0],
            'predicted_goals_away': prediction[0][1],
            'algorithm_used': self.algorithm
        }

# # Example usage:
# if __name__ == "__main__":
#     # Instantiate the PredictOutcome class
#     predictor = PredictOutcome()
#
#     # Train the models and select the best algorithm based on MSE
#     predictor.train_model()
#
#     # Load the trained model
#     predictor.load_model()
#
#     # Sample data for prediction
#     input_data = {
#         'team_home': 'Tokyo Verdy Beleza',
#         'team_away': 'INAC Kobe Leonessa',
#         'possession_home': 53.67,
#         'possession_away': 53.33,
#         'shots_on_target_home': 6.33,
#         'shots_on_target_away': 6
#     }
#
#     # Make prediction
#     result = predictor.predict_scores(input_data)
#     print(result)
