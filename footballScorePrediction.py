import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import joblib


def get_data_prompt(team):
    p = f"in tabular form, list the last 10 games played by {team} with the following information:  " \
        f"team_home, team_away, goals_home,goals_away, possession_home, possession_away, " \
        f"shots_on_target_home,shots_on_target_away"
    print(p)
    return p

def get_predict_prompt(home, away):
    p = f"""there is an upcoming match between {home}(home) and {away}(Away):

based on the last 3 games played by {home}  and the last 3 games played by {away}, 
provide average possession and shots on target for both team using the format below

        'team_home': ['{home}'],'team_away': ['{away}'],'possession_home': [0.47],'possession_away': [0.55],'shots_on_target_home': [3.67],'shots_on_target_away': [4.33]

    """
    print(p)
    return p

def preprocess_data_old(data, label_encoder=None):
    # Convert percentage values to numeric
    # data['possession_home'] = data['possession_home'].str.rstrip('%').astype('float') / 100.0
    # data['possession_away'] = data['possession_away'].str.rstrip('%').astype('float') / 100.0

    # Encode team names using LabelEncoder
    if label_encoder is None:
        label_encoder = LabelEncoder()
        data['team_home_encoded'] = label_encoder.fit_transform(data['team_home'])
        data['team_away_encoded'] = label_encoder.fit_transform(data['team_away'])
    else:
        data['team_home_encoded'] = label_encoder.transform(data['team_home'])
        # Handle new team names during prediction
        data['team_away_encoded'] = label_encoder.transform(data['team_away'])

    return data, label_encoder

def preprocess_data(data, label_encoder=None):
    # Convert percentage values to numeric
    # data['possession_home'] = data['possession_home'].str.rstrip('%').astype('float') / 100.0
    # data['possession_away'] = data['possession_away'].str.rstrip('%').astype('float') / 100.0

    # Encode team names using LabelEncoder
    if label_encoder is None:
        label_encoder = LabelEncoder()

    # Fit the encoder with the union of training and prediction data
    all_teams = pd.concat([data['team_home'], data['team_away']]).unique()
    label_encoder.fit(all_teams)

    data['team_home_encoded'] = label_encoder.transform(data['team_home'])
    data['team_away_encoded'] = label_encoder.transform(data['team_away'])

    return data, label_encoder

def train_model(data):
    # Split the data into features (X) and target variables (y)

    X = data[['team_home_encoded', 'team_away_encoded', 'possession_home', 'possession_away', 'shots_on_target_home', 'shots_on_target_away']]
    y_home = data['goals_home']
    y_away = data['goals_away']

    # Train a linear regression model for home goals
    model_home = LinearRegression()
    model_home.fit(X, y_home)

    # Train a linear regression model for away goals
    model_away = LinearRegression()
    model_away.fit(X, y_away)

    print('training completed!')

    return model_home, model_away

def save_model(model, filename):
    joblib.dump(model, filename)

def load_model(filename):
    return joblib.load(filename)

def predict_scores(model_home, model_away, label_encoder, new_game):
    # Encode team names in the new game
    new_game['team_home_encoded'] = label_encoder.transform(new_game['team_home'])
    new_game['team_away_encoded'] = label_encoder.transform(new_game['team_away'])

    # Extract features for prediction
    X_new_game = new_game[['team_home_encoded', 'team_away_encoded', 'possession_home', 'possession_away', 'shots_on_target_home', 'shots_on_target_away']]

    # Predict scores for the new game
    predicted_goals_home = model_home.predict(X_new_game)
    predicted_goals_away = model_away.predict(X_new_game)

    return predicted_goals_home[0], predicted_goals_away[0]


if __name__ == "__main__":
    # Load the data from the Excel file
    import os
    while True:
        os.system("cls")
        ans = input("1. prompt or 2. predict: ")
        if int(ans) == 2:
            data = pd.read_excel('football_scores_data.xlsx')

            # Preprocess the data and get label encoder
            processed_data, label_encoder = preprocess_data(data)

            # Train the model
            model_home, model_away = train_model(processed_data)
            print("Saving model...")
            # Save the models
            save_model(model_home, 'model_home.joblib')
            save_model(model_away, 'model_away.joblib')
            print("model saved...")

            new_game_data = input("New Game Data:")
            updated = ""
            for c in new_game_data:
                if ord(c) == 8220 or ord(c) == 8221:
                    c = '"'
                if ord(c) == 8216 or ord(c) == 8217:
                    c = '"'
                updated += c

            new_game_data = eval("{" + updated + "}")

            if new_game_data['possession_home'][0] > 1:
                new_game_data['possession_home'] = [new_game_data['possession_home'][0]/100]
                new_game_data['possession_away'] = [new_game_data['possession_away'][0]/100]

            # Example: Predict scores for a new game
            new_game = pd.DataFrame(new_game_data)

            # new_game = pd.DataFrame({
            #     'team_home': ['Osasuna'],
            #     'team_away': ['Real Sociedad'],
            #     'possession_home': [0.47],
            #     'possession_away': [0.55],
            #     'shots_on_target_home': [3.67],
            #     'shots_on_target_away': [4.33]
            # })

            # Predict scores using the loaded models
            loaded_model_home = load_model('model_home.joblib')
            loaded_model_away = load_model('model_away.joblib')
            predicted_goals_home, predicted_goals_away = predict_scores(loaded_model_home, loaded_model_away, label_encoder, new_game)

            # Display the predictions
            print(f"Predicted Goals for Home Team [{new_game['team_home'][0]}]:", round(predicted_goals_home,3))
            print(f"Predicted Goals for Away Team [{new_game['team_away'][0]}]:", round(predicted_goals_away,3))
        elif int(ans) == 1:
            home = input("Home team: ")
            if "," not in home:
                away = input("Away team: ")
            else:
                home, away = home.split(",")

            get_data_prompt(home)
            get_data_prompt(away)

            get_predict_prompt(home,away)
        else:
            break

