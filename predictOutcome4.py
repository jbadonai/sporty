import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
import joblib
import pickle
import os

from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.linear_model import SGDRegressor, ElasticNet
from sklearn.neighbors import RadiusNeighborsRegressor
from sklearn.tree import ExtraTreeRegressor
from sklearn.ensemble import BaggingRegressor

class PredictOutcome:
    def __init__(self, data_file='football_team_training_data.xlsx', parent=None):
        self.data_file = data_file
        self.model = None
        self.model_file = "saved_model.pkl"  # Change the file name as needed
        self.parent = parent


    def load_data(self):
            # Load data from Excel file
            df = pd.read_excel(self.data_file)

            # Drop team_home and team_away columns
            df = df.drop(['team_home', 'team_away'], axis=1)

            # Convert percentage columns to numeric
            def convert_possession(value):
                if isinstance(value, str) and '%' in value:
                    return float(value.rstrip('%')) / 100.0
                else:
                    return float(value) / 100

            df['possession_home'] = df['possession_home'].apply(convert_possession)
            df['possession_away'] = df['possession_away'].apply(convert_possession)

            # df['possession_home'] = pd.to_numeric(df['possession_home'].replace('%', '', regex=True))
            # df['possession_away'] = pd.to_numeric(df['possession_away'].replace('%', '', regex=True))

            # Set features and target
            features = df.drop(['goals_home', 'goals_away'], axis=1)
            target = df[['goals_home', 'goals_away']]

            return features, target

    def split_data(self, features, target):
        # Split data into training and testing sets
        features_train, features_test, target_train, target_test = train_test_split(
            features, target, test_size=0.2, random_state=42
        )

        return features_train, features_test, target_train, target_test

    def train_model(self, features_train, target_train, algorithm):
        # Train a specified regression model
        if isinstance(algorithm, SVR):
            # Combine 'goals_home' and 'goals_away' into a single target variable
            target_combined = target_train['goals_home'] + target_train['goals_away']

            # Train a single SVM model for the combined target variable
            model = algorithm.fit(features_train, target_combined)
            return model
        else:
            # For other algorithms, use the same approach as before
            model = algorithm.fit(features_train, target_train)
            return model

    def evaluate_model(self, model, features_test, target_test):
        # Evaluate the model using Mean Squared Error
        if isinstance(model, SVR):
            # For SVM model, predict combined targets
            predictions = model.predict(features_test)
            mse_home = mean_squared_error(target_test['goals_home'], predictions[:, 0])
            mse_away = mean_squared_error(target_test['goals_away'], predictions[:, 1])
        else:
            # For other models, use the same approach as before
            predictions = model.predict(features_test)
            mse_home = mean_squared_error(target_test['goals_home'], predictions[:, 0])
            mse_away = mean_squared_error(target_test['goals_away'], predictions[:, 1])

        # You might want to return both MSE values or some aggregate measure
        return mse_home, mse_away

    def save_model(self, model):
        # Save the trained model to a file using joblib
        joblib.dump(model, self.model_file)

    def training(self):
        # Step 1: Load data
        features, target = self.load_data()

        # Step 2: Split data
        features_train, features_test, target_train, target_test = self.split_data(features, target)

        # Step 3: Train and evaluate models
        # algorithms = [
        #     ("Linear Regression", LinearRegression()),
        #     ("Random Forest", RandomForestRegressor(random_state=42)),
        #     ("Decision Tree", DecisionTreeRegressor(random_state=42)),
        #     ("K-Nearest Neighbors", KNeighborsRegressor()),
        #     ("Ridge Regression", Ridge()),
        #     ("Neural Network", MLPRegressor(random_state=42)),
        #     ("Lasso Regression", Lasso()),
        #     # Add other algorithms as needed
        # ]

        algorithms = [
            ("Linear Regression", LinearRegression()),
            ("Random Forest", RandomForestRegressor(random_state=42)),
            ("Decision Tree", DecisionTreeRegressor(random_state=42)),
            ("K-Nearest Neighbors", KNeighborsRegressor()),
            ("Ridge Regression", Ridge()),
            ("Neural Network", MLPRegressor(random_state=42)),
            ("Lasso Regression", Lasso()),
            ("Gradient Boosting", GradientBoostingRegressor(random_state=42)),
            ("AdaBoost", AdaBoostRegressor(random_state=42)),
            ("Support Vector Regression", SVR()),
            ("Stochastic Gradient Descent", SGDRegressor(random_state=42)),
            ("Elastic Net", ElasticNet(random_state=42)),
            ("Radius Neighbors", RadiusNeighborsRegressor()),
            ("Extra Trees", ExtraTreeRegressor(random_state=42)),
            ("Bagging Regressor", BaggingRegressor(random_state=42)),
            # Add other algorithms as needed
        ]

        best_model = None
        best_aggregate_mse = float('inf')  # Initialize with a large value

        for name, algorithm in algorithms:
            try:
                # Train model
                model = self.train_model(features_train, target_train, algorithm)

                # Evaluate model
                mse_home, mse_away = self.evaluate_model(model, features_test, target_test)

                # Aggregate the mean squared errors
                aggregate_mse = mse_home + mse_away

                print(f"{name} Mean Squared Error: ({mse_home}, {mse_away})")

                # Update best model if current model is better
                if aggregate_mse < best_aggregate_mse:
                    best_model = model
                    best_aggregate_mse = aggregate_mse
            except:
                continue

        # Step 4: Save the best model
        print()
        print('--------------------------------------')
        print(f"best mode: {best_model}")
        print('--------------------------------------')
        print()
        self.save_model(best_model)
        return best_model

    def load_saved_model(self):
        # Load the saved model from a file using joblib
        self.model = joblib.load(self.model_file)
        print(f"Model loaded successfully.")

    def predicting(self, features_input):
        # Step 5: Load the saved model
        if self.model is None:
            raise ValueError("Model not loaded. Call load_saved_model() first.")

        # Step 6: Make predictions
        if isinstance(self.model, SVR):
            # For SVM model, predict combined targets
            predictions = self.model.predict(features_input)
        else:
            # For other models, use the same approach as before
            predictions = self.model.predict(features_input)

        # Format predictions as Home X : Away Y
        home_goals = int(round(predictions[0][0]))
        away_goals = int(round(predictions[0][1]))

        result = f"Home {home_goals} : Away {away_goals}"
        return result

#
# # Example usage:
# predictor = PredictOutcome()
#
# if os.path.exists(predictor.model_file) is False:
#     predictor.training()
#     print("Training completed successfully!")
#
#
# # Assuming features_input is a DataFrame with the same columns as the training data
# features_input = pd.DataFrame({
#     'possession_home': [10],
#     'possession_away': [52],
#     'shots_on_target_home': [1],
#     'shots_on_target_away': [6]
# })
#
# # Convert percentage columns to numeric
#
#
# def convert_possession(value):
#     if isinstance(value, str) and '%' in value:
#         return float(value.rstrip('%')) / 100.0
#     else:
#         return float(value) / 100
#
#
# features_input['possession_home'] = features_input['possession_home'].apply(convert_possession)
# features_input['possession_away'] = features_input['possession_away'].apply(convert_possession)
#
# # features_input['possession_home'] = pd.to_numeric(features_input['possession_home'].replace('%', '', regex=True))
# # features_input['possession_away'] = pd.to_numeric(features_input['possession_away'].replace('%', '', regex=True))
#
# predictor.load_saved_model()  # Load the saved model
#
#
# result = predictor.predicting(features_input)
# print(result)
