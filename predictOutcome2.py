import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

def clean_data(data):
    # Cleaning possession data
    # data['possession_home'] = data['possession_home'].str.rstrip('%').astype('float') / 100.0
    # data['possession_away'] = data['possession_away'].str.rstrip('%').astype('float') / 100.0

    def convert_possession(value):
        if isinstance(value, str) and '%' in value:
            return float(value.rstrip('%')) / 100.0
        else:
            return float(value) / 100

    data['possession_home'] = data['possession_home'].apply(convert_possession)
    data['possession_away'] = data['possession_away'].apply(convert_possession)

    # Drop 'team_home' and 'team_away' columns
    data = data.drop(['team_home', 'team_away'], axis=1)

    return data

def train_model(data):

    # Feature columns
    feature_columns = ['possession_home', 'possession_away', 'shots_on_target_home', 'shots_on_target_away']

    # Target columns
    target_columns = ['goals_home', 'goals_away']
    # print(data.columns)

    X = data[feature_columns]
    y = data[target_columns]

    # Initialize the RandomForestRegressor
    model = RandomForestRegressor()

    # Train the model
    model.fit(X, y)

    # Save the trained model to a file
    joblib.dump(model, 'football_score_predictor_model.pkl')

def predict_score(game_stat):
    # Load the trained model
    model = joblib.load('football_score_predictor_model.pkl')

    # Create a DataFrame for prediction
    data = pd.DataFrame({
        'possession_home': [game_stat['possession_home']],
        'possession_away': [game_stat['possession_away']],
        'shots_on_target_home': [game_stat['shots_on_target_home']],
        'shots_on_target_away': [game_stat['shots_on_target_away']]
    })
    # print(data.columns)
    # print(data)

    # Make prediction
    predicted_scores = model.predict(data)[0]
    home_predicted_score = int(predicted_scores[0])
    away_predicted_score = int(predicted_scores[1])

    return home_predicted_score, away_predicted_score

# if __name__ == "__main__":
def start(game_stat=None):
    # Load the training data
    print("V2: loading data from excel file...")
    data = pd.read_excel('football_team_training_data.xlsx')

    # Clean the data
    print("V2: Initial data cleaning...")
    cleaned_data = clean_data(data)

    # Train the model
    print('V2: training model')
    train_model(cleaned_data)

    # Example of predicting scores
#     game_stat = {
#     'team_home': ['Spezia Calcio'],
#     'team_away': ['Ascoli Calcio 1898 FC'],
#     'possession_home': [49],
#     'possession_away': [48],
#     'shots_on_target_home': [4],
#     'shots_on_target_away': [3]
# }
#     print(game_stat)

    # Remove '[' and ']' from the list values
    print('V2: cleaning data')
    for key, value in game_stat.items():
        if isinstance(value, list) and len(value) > 0:
            game_stat[key] = value[0]

    # Store team_home and team_away values in variables
    home = game_stat.pop('team_home', None)
    away = game_stat.pop('team_away', None)

    print("V2: predicting scores...")
    # print(game_stat)
    # game_stat = str(game_stat)
    home_score, away_score = predict_score(game_stat)

    # result = f'Predicted Score: {home}(Home): {home_score} - {away}(Away): {away_score}'
    # print(home_score,"-", away_score)
    # print(home,"-", away)

    return home, home_score, away, away_score


# start()