import sys
from PyQt5.QtWidgets import QMessageBox, QApplication, QWidget, QVBoxLayout, QGroupBox, QLabel, QLineEdit, QPushButton, \
    QTextEdit, QHBoxLayout, QVBoxLayout, QCheckBox
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from joblib import dump
from joblib import load
import re
import pandas as pd
from io import StringIO
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import os
from promptGenerator import GeneratePrompt

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import joblib


class JbaPredictWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("JBA Predict")
        self.setGeometry(100, 100, 800, 400)

        self.model_filename = 'football_pm.joblib'

        self.training_df = None
        self.training_data_filename = "football_team_training_data.xlsx"
        self.prompt_generator = None

        self.sport = "football"
        self.game_period = 6

        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()

        # Top Section - Teams and Prompt Generation
        top_layout = QHBoxLayout()

        # GroupBox for Teams
        teams_groupbox = QGroupBox("Generate Prompts: (Required to get Training and Predict data from AI like Bing...")
        teams_layout = QHBoxLayout()

        # Checkbox for Auto
        self.auto_checkbox = QCheckBox("Auto")
        self.auto_checkbox.stateChanged.connect(self.toggle_auto_mode)

        self.home_team_input_teams = QLineEdit()
        self.away_team_input_teams = QLineEdit()
        self.league_input = QLineEdit()
        self.sport_input = QLineEdit()
        self.game_period_input = QLineEdit()

        self.home_team_input_teams.setPlaceholderText("Home Team")
        self.away_team_input_teams.setPlaceholderText("Away Team")
        self.league_input.setPlaceholderText("League")
        self.sport_input.setPlaceholderText("Sport")
        self.game_period_input.setPlaceholderText("Game Period")

        self.sport_input.setText("football")
        self.game_period_input.setText("6")

        get_prompts_button = QPushButton("Get Prompts")
        get_prompts_button.clicked.connect(self.get_prompts)

        teams_layout.addWidget(self.auto_checkbox)
        teams_layout.addWidget(self.home_team_input_teams)
        teams_layout.addWidget(self.away_team_input_teams)
        teams_layout.addWidget(self.league_input)
        teams_layout.addWidget(self.sport_input)
        teams_layout.addWidget(self.game_period_input)
        teams_layout.addWidget(get_prompts_button)

        teams_groupbox.setLayout(teams_layout)
        top_layout.addWidget(teams_groupbox)

        main_layout.addLayout(top_layout)

        # used to arrange the button horizontally
        buttons_groupbox = QGroupBox("")
        buttons_layout = QHBoxLayout()

        # Bottom Section - Prediction
        prediction_layout = QVBoxLayout()

        prompt_code_label = QLabel("Training Data / Predict Data:")
        self.prompt_code_input = QTextEdit()

        self.predict_button = QPushButton("Predict")
        # predict_button.clicked.connect(self.predict)
        self.predict_button.clicked.connect(self.start)

        self.load_train_data_button = QPushButton("Load Training Data")
        self.load_train_data_button.clicked.connect(self.load_training_data)

        prediction_result_label = QLabel("Prediction Result:")
        self.prediction_result_display = QTextEdit()
        self.prediction_result_display.setReadOnly(True)
        text_edit_font = QFont()
        text_edit_font.setPointSize(10)  # Set the font size to 14
        self.prediction_result_display.setFont(text_edit_font)

        buttons_layout.addWidget(self.load_train_data_button)
        buttons_layout.addWidget(self.predict_button)

        buttons_groupbox.setLayout(buttons_layout)

        prediction_layout.addWidget(prompt_code_label)
        prediction_layout.addWidget(self.prompt_code_input)
        # prediction_layout.addWidget(self.load_train_data_button)
        # prediction_layout.addWidget(self.predict_button)

        prediction_layout.addWidget(buttons_groupbox)
        prediction_layout.addWidget(prediction_result_label)
        prediction_layout.addWidget(self.prediction_result_display)

        self.prompt_code_input.textChanged.connect(self.clear_prediction)

        # Add Prediction section to the main layout
        main_layout.addLayout(prediction_layout)

        self.setLayout(main_layout)

        # self.predict_button.setVisible(False)
        # self.load_train_data_button.setVisible(True)

    def toggle_auto_mode(self, state):
        if state == Qt.Checked:
            # Auto mode is checked, disable other components
            self.home_team_input_teams.setEnabled(False)
            self.away_team_input_teams.setEnabled(False)
            self.league_input.setEnabled(False)
            # Add more components if needed
        else:
            # Auto mode is unchecked, enable other components
            self.home_team_input_teams.setEnabled(True)
            self.away_team_input_teams.setEnabled(True)
            self.league_input.setEnabled(True)
            # Add more components if needed

    def get_prompts(self):

        if self.auto_checkbox.isChecked():
            self.sport = self.sport_input.text()
            self.game_period = self.game_period_input.text()
            self.prompt_generator = GeneratePrompt(sport=self.sport, period=self.game_period)
            self.prompt_generator.start()
            pass
        else:
            home_team = self.home_team_input_teams.text()
            away_team = self.away_team_input_teams.text()
            league = self.league_input.text()
        # Add your logic here to use home_team and away_team to generate prompts
        # Update the UI or perform other actions as needed

    def clear_prediction(self):
        self.prediction_result_display.clear()

    def save_model_f(self, model):
        # Save the model to a file
        dump(model, self.model_filename)

    def load_model_f(self):
        # Load the saved model
        loaded_model = load(self.model_filename)
        # # Use the loaded model for predictions
        # predictions = loaded_model.predict(X_test_scaled)

        return loaded_model

    def preprocess_possession_percentage(self, percentage_str):
        return float(percentage_str.replace('%', ''))

    def predict_match_outcome(self, is_home, recent_performance, average_goals, player_injuries,
                              possession_percentage, strength, head_to_head, opponent_is_home, opponent_recent_performance,
                              opponent_average_goals, opponent_player_injuries,opponent_possession_percentage,
                              opponent_strength, opponent_head_to_head):
        try:
            # Load the saved model
            loaded_model = load(self.model_filename)

            input_data = pd.DataFrame({
                'Is Home': [is_home],
                'Recent 5 Performance': [recent_performance],
                'Average Goals': [average_goals],
                'Player Injuries': [player_injuries],
                'Possession Percentage': [str(possession_percentage)],
                'Strength': [strength],
                'Head-to-Head': [head_to_head],
                'Opponent Is Home': [opponent_is_home],
                'Opponent Recent 5 Performance': [opponent_recent_performance],
                'Opponent Average Goals': [opponent_average_goals],
                'Opponent Player Injuries': [opponent_player_injuries],
                'Opponent Possession Percentage': [str(opponent_possession_percentage)],
                'Opponent Strength': [opponent_strength],
                'Opponent Head-to-Head': [opponent_head_to_head],
            })



            input_data[['Team_Wins_recent', 'Team_Losses_recent', 'Team_Draws_recent']] = input_data['Recent 5 Performance'].astype(
                str).str.split(':', expand=True).astype(int)
            input_data[['Team_Wins_head2head', 'Team_Losses_head2head', 'Team_Draws_head2head']] = input_data['Head-to-Head'].astype(
                str).str.split(':', expand=True).astype(int)

            input_data[['opp_Wins_recent', 'opp_Losses_recent', 'opp_Draws_recent']] = input_data['Opponent Recent 5 Performance'].astype(
                str).str.split(':', expand=True).astype(int)
            input_data[['opp_Wins_head2head', 'opp_Losses_head2head', 'opp_Draws_head2head']] = input_data['Opponent Head-to-Head'].astype(
                str).str.split(':', expand=True).astype(int)



            # Drop unnecessary columns
            input_data = input_data.drop(['Recent 5 Performance', 'Head-to-Head','Opponent Recent 5 Performance',
                                          'Opponent Head-to-Head'], axis=1)

            # Convert 'Is Home' column to binary (1 for True, 0 for False)
            input_data['Is Home'] = input_data['Is Home'].astype(int)

            # Preprocess 'Possession Percentage' column
            input_data['Possession Percentage'] = input_data['Possession Percentage'].apply(self.preprocess_possession_percentage)
            input_data['Opponent Possession Percentage'] = input_data['Opponent Possession Percentage'].apply(self.preprocess_possession_percentage)

            # Make predictions using the loaded model
            prediction = loaded_model.predict(input_data)

            # Print the prediction
            return "Home Win" if prediction == 1 else "Away Win" if prediction == 0 else "Draw"
            # return f'Team: {team_name}\nPredicted Outcome: {"Home Win" if prediction == 1 else "Away Win" if prediction == 0 else "Draw"}'
        except Exception as e:
            print(f"An Error occurred in 'predict match outcome: : {e}")

    def train_model_f(self):
        try:
            # Load the data from the Excel file
            file_path = 'training_data.xlsx'

            if os.path.exists(file_path) is False:
                QMessageBox.critical(self, "Missing Data!",
                                     "Training data is missing. 'league data.xlsx' file missing in the root directory")
                raise Exception
            df = pd.read_excel(file_path)
            # print(df.columns)
            # input('waits')

            # Convert 'Recent 5 Performance' and 'Head-to-Head' columns to the correct format
            # Home
            print(1)
            df[['Home_Wins_recent', 'Home_Losses_recent', 'Home_Draws_recent']] = df['Team Recent 5 Performance'].astype(str).str.split(':',expand=True).astype(int)
            df[['Home_Wins_head2head', 'Home_Losses_head2head', 'Home_Draws_head2head']] = df['Team Head-to-Head'].astype(str).str.split(':', expand=True).astype(int)

            # Opponent
            print(2)
            df[['Opp_Wins_recent', 'Opp_Losses_recent', 'Opp_Draws_recent']] = df['Opponent Recent 5 Performance'].astype(str).str.split(':',expand=True).astype(int)
            df[['Opp_Wins_head2head', 'Opp_Losses_head2head', 'Opp_Draws_head2head']] = df['Opponent Head-to-Head'].astype(str).str.split(':', expand=True).astype(int)

            # [old] Drop unnecessary columns
            # [old] df = df.drop(['Team Name', 'Recent 5 Performance', 'Head-to-Head'], axis=1)
            print(3)
            df = df.drop(['Team Name', 'Team Recent 5 Performance', 'Team Head-to-Head', 'Opponent Team Name',
                          'Opponent Recent 5 Performance', 'Opponent Head-to-Head'], axis=1)

            # Convert 'Team is Home' and 'Opponent is Home' columns to binary (1 for True, 0 for False)
            df['Team is Home'] = df['Team is Home'].astype(int)
            df['Opponent is Home'] = df['Opponent is Home'].astype(int)
            print(4)
            # Map 'Match Outcome' to binary (1 for Home Win, 0 for Away Win, 2 for Draw)
            df['Match Outcome'] = df['Match Outcome'].map({'Home Win': 1, 'Away Win': 0, 'Draw': 2})

            # Split the data into features (X) and target (y)
            X = df.drop('Match Outcome', axis=1)
            y = df['Match Outcome']

            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

            # Standardize the features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # --------------------------------------
            models = {
                'Logistic Regression': LogisticRegression(random_state=42),
                'Random Forest': RandomForestClassifier(random_state=42),
                'Support Vector Machine': SVC(random_state=42),
                'Gradient Boosting': GradientBoostingClassifier(random_state=42)
            }

            # Iterate through the models and evaluate their performance
            highest_accuracy = 0
            selected_model = ""
            selected_model_name = ""
            for model_name, model in models.items():
                # Train the model
                model.fit(X_train_scaled, y_train)

                # Make predictions on the test set
                y_pred = model.predict(X_test_scaled)

                # Evaluate the accuracy of the model
                accuracy = accuracy_score(y_test, y_pred)
                if accuracy > highest_accuracy:
                    highest_accuracy = accuracy
                    selected_model = model
                    selected_model_name = model_name

                print(f'{model_name} Accuracy: {accuracy * 100:.2f}%')

            self.save_model(selected_model)
            print(f'[ SAVED ] | {selected_model_name} Accuracy: {highest_accuracy * 100:.2f}%')
        except Exception as e:
            print(f"An error occurred in train model: {e}")

        # ------------------------------------

    def generate_prompt(self):
        sport = self.sport_input.text()
        league = self.league_input.text()
        home_team = self.home_team_input.text()
        away_team = self.away_team_input.text()

        # Add your prompt generation logic here
        # For now, just display the input in the result label

        prompt = f"""
        In an upcoming {sport} match, in {league} league, between {home_team} (home) vs {away_team} (away):

        * opponent_strength will be statistics for {away_team} (away) while the rest will be statistics for {home_team} (home)  

        get the following data:
        team name,is home(True if home), recent 5 performance (in format w:l:d eg 2:0:1,  Don't use '-' delimeter),  average_goals, player_injuries(use 0 if none), possession_percentage, head_to_head (in format w:l:d eg 2:0:1, Don't use '-' delimeter), opponent_strength (to be calculated using Elo rating))

        Present the data in the format below:
        team name__is home__recent 5 performance__average goals__player injuries__possession percentage__head to head__opponent strengt
            """

        self.result_label.setText(prompt)

    def copy_result(self):
        result_text = self.result_label.text()
        QApplication.clipboard().setText(result_text)

    def predict_old(self):
        try:
            # holds error to be displayed
            internal_error = ""

            # get the prompt code
            prompt_code = self.prompt_code_input.toPlainText().replace("#", '"')
            prompt_code = eval(prompt_code)



            # check if trained model is available, if not train a model
            if os.path.exists(self.model_filename) is False:
                print("No existing model Found! Training model. Please wait...")
                QMessageBox.information(self, "No Model!", "Model will be trainied now. Please wait.")
                self.train_model()

            print("Predicting Outcome!")

            final_prediction = []
            teams = []

            # extract features from prompt code
            home_team_name = prompt_code[0][1]
            home_is_home = prompt_code[1][1]
            home_recent_5_performance = prompt_code[2][1]
            home_average_goals = prompt_code[3][1]
            home_player_injuries = prompt_code[4][1]
            home_possession_percentage = str(prompt_code[5][1])
            home_strenght = prompt_code[6][1]
            home_head_to_head = prompt_code[7][1]
            away_team_name = prompt_code[8][1]
            away_is_home = prompt_code[9][1]
            away_recent_5_performance = prompt_code[10][1]
            away_average_goals = prompt_code[11][1]
            away_player_injuries = prompt_code[12][1]
            away_possession_percentage = str(prompt_code[13][1])
            away_strenght = prompt_code[14][1]
            away_head_to_head = prompt_code[15][1]

            result = self.predict_match_outcome(home_is_home, home_recent_5_performance, home_average_goals,
            home_player_injuries, home_possession_percentage, home_strenght, home_head_to_head,
            away_is_home, away_recent_5_performance, away_average_goals, away_player_injuries,
            away_possession_percentage, away_strenght, away_head_to_head)

            league, clock = self.find_league(home_team_name, away_team_name)
            league = league.split(' ')[1:]
            final_result = f"[ LEAGUE ]: \t{' '.join(league)}  \n[ TIME ]: \t{clock} \n[ TEAM ]: \t{home_team_name}  vs  {away_team_name} \n\n[ PREDICTION ]: \t{result.upper()}"

            self.prediction_result_display.setPlainText(f"{final_result}")
            self.prompt_code_input.clear()
        except Exception as e:
            if internal_error != "":
                print(f"An Error Occurred! {internal_error}")
                self.prediction_result_display.setPlainText(f"An Error Occurred! {internal_error}")
            else:
                print(f"An Error Occurred in 'predict': {e}")
                self.prediction_result_display.setPlainText(f"An Error Occurred in 'predict': {e}")

    def pre_process(self, data):
        final_data = ""
        start = False
        end = False
        for c in data:
            if ord(c) == 8220 or ord(c) == 8221:
                c = '"'

            if ord(c) == 8216 or ord(c) == 8217:
                c = '"'


            if start is False and c=="[" :
                start = True
                end = False

            if start is True and c== "]":
                start = False
                end = True

            if start is True:
                final_data += c

            if end is True:
                final_data += f"{c};"
                end = False

        return final_data[:-1]

    def pre_process_no_quote(self, data_string):

        def add_quotes(match):
            first_group = match.group(1)
            second_group = match.group(2)

            # Check if the second group is a number (int, float) or a boolean (True, False)
            if re.match(r'^-?\d+(?:\.\d+)?$', second_group) or second_group.lower() in ('true', 'false'):
                return f"('{first_group}', {second_group})"
            else:
                return f"('{first_group}', '{second_group}')"

        # Use regular expression to find and modify text elements
        modified_data_string = re.sub(r'\(([^,]+),\s*([^),]+)\)', add_quotes, data_string)

        return modified_data_string

    def get_character_code(character):
       return ord(character)

    def predict(self):
        try:
            prompt_code = self.prompt_code_input.toPlainText()

            prompt_code = self.pre_process(prompt_code)

            if prompt_code.__contains__("#"):
                prompt_code = prompt_code.replace("#", '"')

            if "'" not in prompt_code and '"' not in prompt_code:
                prompt_code = self.pre_process_no_quote(prompt_code)

            data_list = str(prompt_code).split(';')
            # print(f'total: {len(data_list)}')
            # input('wait')
            prediction_list = []
            for data in data_list:
                try:
                    # print(data)
                    prompt_code = eval(data)
                    # print(type(prompt_code))

                    # check if trained model is available, if not train a model
                    if os.path.exists(self.model_filename) is False:
                        print("No existing model Found! Training model. Please wait...")
                        QMessageBox.information(self, "No Model!", "Model will be trainied now. Please wait.")
                        self.train_model()

                    print("Predicting Outcome!")

                    final_prediction = []
                    teams = []

                    # extract features from prompt code
                    home_team_name = prompt_code[0][1]
                    home_is_home = prompt_code[1][1]
                    home_recent_5_performance = prompt_code[2][1]
                    home_average_goals = prompt_code[3][1]
                    home_player_injuries = prompt_code[4][1]
                    home_possession_percentage = str(prompt_code[5][1])
                    home_strenght = prompt_code[6][1]
                    home_head_to_head = prompt_code[7][1]
                    away_team_name = prompt_code[8][1]
                    away_is_home = prompt_code[9][1]
                    away_recent_5_performance = prompt_code[10][1]
                    away_average_goals = prompt_code[11][1]
                    away_player_injuries = prompt_code[12][1]
                    away_possession_percentage = str(prompt_code[13][1])
                    away_strenght = prompt_code[14][1]
                    away_head_to_head = prompt_code[15][1]

                    result = self.predict_match_outcome(home_is_home, home_recent_5_performance, home_average_goals,
                                                        home_player_injuries, home_possession_percentage, home_strenght,
                                                        home_head_to_head,
                                                        away_is_home, away_recent_5_performance, away_average_goals,
                                                        away_player_injuries,
                                                        away_possession_percentage, away_strenght, away_head_to_head)

                    league, clock = self.find_league(home_team_name, away_team_name)
                    league = league.split(' ')[1:]
                    odds = self.find_odds(home_team_name, away_team_name)

                    h = float(odds[0])
                    d = float(odds[1])
                    a = float(odds[2])

                    odd_diff = round(abs(h-a), 1)
                    secondOpinion = self.second_opinion(h,d,a,result)

                    final_result = f"[ LEAGUE ]: \t{' '.join(league)}  \n[ TIME ]: \t{clock} \n[ TEAM ]: \t{home_team_name}  vs  {away_team_name} \n" \
                        f"[Odds] : \t H: {h} | D: {d} | A: {a} || diff: [ {odd_diff} ]\n\n" \
                        f"[ PREDICTION ]: \t\t{result.upper()}\n" \
                        f"[ MARKET ODD ]: \t{secondOpinion}"

                    prediction_list.append(final_result)
                    pass
                except Exception as e:
                    print(f">>> ERROR!!!!!!! \n{e}")
                    self.prediction_result_display.setPlainText(f">>> ERROR!!!!!!! \n{e}")
                    continue
        except Exception as e:
            print(f"An Error Occurred in predict: {e}")

        predictionText = ""
        for p in prediction_list:
            predictionText = predictionText + p + "\n\n" + "*" * 80 + "\n\n"

        self.prompt_code_input.clear()
        self.prediction_result_display.setPlainText(f"{predictionText}")

    def find_highest(self, hd, da, ha):
        # Find the highest number
        highest = None
        chosen = ""
        abrv = ""

        if hd >= da and hd >= ha:
            highest = hd
            chosen = "Home or Draw"
            abrv = 'hd'
        elif da >= hd and da >= ha:
            highest = da
            chosen = "Away or Draw"
            abrv = 'da'
        else:
            highest = ha
            chosen = "Home or Away"
            abrv = 'ha'

        return chosen, highest, abrv

    def probaiblity(self, h, d, a):
        # Convert Percentages to Probabilities:
        # a:home odd %, b:draw odd %, c:away odd %
        total = h + d + a
        ph = h/total
        pd = d/total
        pa = a/total

        # Calculate Combined Probabilities:   # Convert Combined Probabilities to Percentages (Optional):
        phd = round(ph * pd * 100, 2)
        pda = round(pd * pa * 100, 2)
        pha = round(ph * pa * 100, 2)

        # find the highest probaility
        odd_predict, odd_percentage, abrv = self.find_highest(phd, pda, pha)

        return odd_predict, f"{odd_percentage}% <||> [ hd:{phd} | ad:{pda} | ha:{pha} ]"

    def second_opinion(self, home_odd, draw_odd, away_odd, predicted):
        # determining result using market odds

        if home_odd == 0 or away_odd == 0:
            return "Odd Info Not Available! Cannot give second opinion!"
        opinion = ""
        odd_diff = round(abs(home_odd - away_odd), 1)

        implied_probaility = (round((1/home_odd) * 100, 2),round((1/draw_odd)* 100, 2),round((1/away_odd)* 100, 2))

        print(f"implied probaility: {implied_probaility}")

        odd_predict, odd_percentage = self.probaiblity(implied_probaility[0], implied_probaility[1], implied_probaility[2])

        # hap = implied_probaility[0] + implied_probaility[2]
        # hdp = implied_probaility[0] + implied_probaility[1]
        # adp = implied_probaility[1] + implied_probaility[2]
        #
        # odd_predict, odd_percentage = self.find_highest(hap, hdp, adp)

        # if odd_diff < 0.5:
        #     opinion = f"{predicted} Or Draw"
        #     pass
        # elif odd_diff >= 0.5 and odd_diff <=1:
        #     opinion = f"{predicted.split(' ')[0]} Or Draw"
        #     pass
        # elif odd_diff > 1 and odd_diff <= 5:
        #     opinion = "Home or Away"
        #     pass
        # elif odd_diff > 5 and odd_diff <= 10:
        #     opinion = "Home or Away"
        #     pass
        # else:
        #     if predicted.lower().__contains__('win'):
        #         if home_odd < away_odd:
        #             opinion = "Home"
        #         else:
        #             opinion = "Away"
        #
        #         # opinion = predicted.split(" ")[0]
        #     pass

        final_opinion = f"{odd_predict} - {odd_percentage}"

        return final_opinion

    def find_league(self, home_team, away_team):
        if os.path.exists("leagueTable.txt") is False:
            return "Unknown - League Table not Found! Run Prompt Generator Again!"
        with open("leagueTable.txt", 'r') as f:
            data = f.read()

        league_table = eval(data)

        for home, away, league, clock in league_table:
            if str(home).lower().strip() == str(home_team).lower().strip() and str(away).lower().strip() == str(away_team).lower().strip():
                return league, clock
        return "Unknown League Name not found in the League Table", "-.--"  # Return None if the match is not found

    def find_odds(self, home_team, away_team):
        if os.path.exists("odds.txt") is False:
            return "Unknown - odds Table not Found! Run Prompt Generator Again!"
        with open("odds.txt", 'r') as f:
            data = f.read()

        odd_table = eval(data)

        for home, away, odds in odd_table:
            if str(home).lower().strip() == str(home_team).lower().strip() and str(away).lower().strip() == str(away_team).lower().strip():
                return odds
        return [0,0,0]  # Return None if the match is not found

    # CODE FROM 'FOOTBALSCOREPREDICTION' BEGINS HERE
    # ======================================================

    def save_dataframe(self, df):
        try:
            # Try to read the existing data from the Excel file
            existing_df = pd.read_excel(self.training_data_filename, sheet_name='Sheet1')

            # Check if the new DataFrame is a subset of the existing DataFrame
            if df.isin(existing_df.to_dict(orient='list')).all().all():
                QMessageBox.information(self, "Data Already Existing!", "Data provided is already in the training data. Skipping duplicate data which might affect prediction")
                print('DataFrame is already a subset of the existing data. Skipping...')
            else:
                # Concatenate the existing data with the new data
                combined_df = pd.concat([existing_df, df], ignore_index=True)

                # Write the combined DataFrame back to the Excel file
                combined_df.to_excel(self.training_data_filename, index=False, sheet_name='Sheet1')
                print(f'DataFrame has been appended to {self.training_data_filename}')
        except FileNotFoundError:
            # If the file doesn't exist, create a new Excel file with the DataFrame
            df.to_excel(self.training_data_filename, index=False, sheet_name="Sheet1")
            print(f'New DataFrame has been written to {self.training_data_filename}')

    def load_training_data(self):
        error_status = ""
        try:

            data = self.prompt_code_input.toPlainText().strip()
            if data == "":
                QMessageBox.information(self, "No Data!", "Please provide Training Data.")
                raise Exception

            counter = 0
            final_data = ""
            line_data = ""
            impurity = False
            dateData = False

            if data.lower().__contains__('date'):
                dataList = data.split("\n")[1:]
                dateData = True
                if str(data.split("\n")[0]).lower().__contains__('last') is False:
                    QMessageBox.information(self, "Wrong info!", "Wrong Training Data detected!\n Team name not found on the firs Line.")
                    self.prompt_code_input.clear()
                    raise Exception

                title = data.split("\n")[0].split("Last")[0]
                title = title.replace("-", "").replace(":", "").replace("'s", "").strip()
                # print(f"title: {title}")
            else:
                dataList = data.split("\n")
                dateData = False
                a = data.count("\n")
                if a < 3 or (data.split("\n")[0] != 'team_home' and data.split("\n")[0].lower() != 'team name'and data.split("\n")[0].lower().__contains__('team name') is False):
                    QMessageBox.information(self, "Wrong info!", "Wrong Training Data detected!")
                    self.prompt_code_input.clear()
                    raise Exception

            for d in dataList:
                if d=="":
                    continue

                counter += 1
                line_data += d + ","
                if d.strip() == "-" or d.strip() == "?" or d.strip() == "TBD":
                    impurity = True

                if counter == 8:
                    counter = 0
                    line_data = line_data[:-1]
                    if impurity is False:
                        final_data += line_data + "\n"
                    else:
                        impurity = False
                    line_data = ""

            # print(final_data)
            # input('::')

            # Read the data from QTextEdit
            data = StringIO(final_data)

            # Read the data into a pandas DataFrame with a comma as a delimiter
            df = pd.read_csv(data, delimiter=',')
            # print(df)
            # input("::")

            header = ["team_home", "team_away", "goals_home", "goals_away", "possession_home", "possession_away",
                      "shots_on_target_home", "shots_on_target_away"]

            # print(df.columns)
            df.columns = header

            if dateData is True:
                df['team_home'] = title

            # Display the resulting DataFrame
            self.save_dataframe(df)
            self.prompt_code_input.clear()

        except Exception as e:
            if error_status != "":
                pass
            print(f" An error occurred in load training data: {e}")
            pass
        pass

    def start(self):
        try:
            # if self.training_df is None:
            #     QMessageBox.information(self, "No Training Data!", "No Training Data detected! Load training "
            #                                                        "data first!")
            #     self.load_train_data_button.setVisible(True)
            #     self.predict_button.setVisible(False)
            #     raise Exception

            print("Training model....")
            predict_prompt = self.prompt_code_input.toPlainText()
            if predict_prompt == "":
                QMessageBox.information(self, "No Team Info", "Team info to predict is required! Please "
                                                                 "provide team info in the required format")
                raise Exception

            a = predict_prompt.count(",")
            if a != 5 :
                QMessageBox.information(self, "Wrong info!", "Wrong Predict prompt detected!")
                self.prompt_code_input.clear()
                raise Exception

            # print(self.training_data_filename)
            data = pd.read_excel(self.training_data_filename)

            # Preprocess the data and get label encoder
            processed_data, label_encoder = self.preprocess_data(data)
            print('ok process data ok.')

            # Train the model
            model_home, model_away = self.train_model(processed_data)
            print("Saving model...")
            # Save the models
            self.save_model(model_home, 'model_home.joblib')
            self.save_model(model_away, 'model_away.joblib')
            print("model saved...")

            new_game_data = predict_prompt
            updated = ""
            for c in new_game_data:
                if c == '%':
                    # QMessageBox.information(self, 'found', 'found')
                    continue
                if ord(c) == 8220 or ord(c) == 8221:
                    c = '"'
                if ord(c) == 8216 or ord(c) == 8217:
                    c = '"'
                updated += c


            if updated.__contains__("{") is False:
                new_game_data = "{" + updated + "}"

            # print(updated)

            new_game_data = eval(updated)
            # print(new_game_data)
            # print(type(new_game_data))

            if new_game_data['possession_home'][0] > 1:
                new_game_data['possession_home'] = [new_game_data['possession_home'][0] / 100]
                new_game_data['possession_away'] = [new_game_data['possession_away'][0] / 100]

            # Example: Predict scores for a new game
            new_game = pd.DataFrame(new_game_data)
            print("loading Model...")

            loaded_model_home = self.load_model('model_home.joblib')
            loaded_model_away = self.load_model('model_away.joblib')

            print("Model Loaded...")
            print("Predicting...")
            predicted_goals_home, predicted_goals_away = self.predict_scores(loaded_model_home, loaded_model_away,
                                                                        label_encoder, new_game)
            print("Done predicting ")
            # Display the predictions

            home_team_name = new_game['team_home'][0]
            away_team_name = new_game['team_away'][0]
            league, clock = self.find_league(home_team_name, away_team_name)
            # league = league.split(' ')[1:]
            odds = self.find_odds(home_team_name, away_team_name)

            diff = round(abs(float(predicted_goals_home)), 1) - round(abs(float(predicted_goals_away)), 1)

            decided = self.decide(predicted_goals_home, predicted_goals_away)

            result = f"""
            [{clock}] - League: {league} --> [{round(abs(diff),1)}]       
            Predicted Goals for Home Team [{new_game['team_home'][0]}]:  {round(predicted_goals_home, 3) } - [{round(predicted_goals_home, 1)}]
            Predicted Goals for Away Team [{new_game['team_away'][0]}]:  {round(predicted_goals_away, 3) } - [{round(predicted_goals_away, 1)}]
            [PLAY]: {decided}
            """


            print(f"Predicted Goals for Home Team [{new_game['team_home'][0]}]:", round(predicted_goals_home, 3))
            print(f"Predicted Goals for Away Team [{new_game['team_away'][0]}]:", round(predicted_goals_away, 3))

            self.prompt_code_input.clear()
            self.prediction_result_display.setText(result)
            self.write_prediction('predictions.txt', result)

            pass
        except Exception as e:
            self.prediction_result_display.setText(str(e))
            if str(e).__contains__('unseen labels'):
                t = str(e).split(":")[-1]
                QMessageBox.information(self, "Missing Data!", f"Missing data for  team {t} in the training data. Previous 10 game statistics for {t} is required in the training data, for successful prediction. \n"
                f"Or Check that the team name is not abrreviated or shortened in the training data. \n"
                f"This error can also occurr if for example you have 'Abu Salim SC' in training data but you are tryinig to predict score for 'Abu Salim' without the 'SC'. ")
            print(f"An Error occurred in start: {e}")

    def decide(self, home_score, away_score):
        diff = abs(home_score - away_score)
        highest = None
        if abs(home_score) > abs(away_score):
            highest = "Home"
        else:
            highest = 'Away'

        if 0 < diff <=1:
            if highest == "Home":
                return f"{highest} or Draw"
            else:
                return f"Draw or {highest}"

        elif 1 < diff <=2:
            return "Home or Away"

        elif diff > 2:
            return highest
        else:
            return "Can't Decide"

    def write_prediction(self, filename, prediction):
        try:
            with open(filename, 'a', encoding='utf-8', errors='replace') as f:
                f.write(prediction)
                f.write("\n")
                f.write('-' * 50)
                f.write("\n")
        except Exception as e:
            print("error in write action..............")

    def get_data_prompt(self, team):
        p = f"in tabular form, list the last 10 games played by {team} with the following information:  " \
            f"team_home, team_away, goals_home,goals_away, possession_home, possession_away, " \
            f"shots_on_target_home,shots_on_target_away"
        print(p)
        return p

    def get_predict_prompt(self, home, away):
        p = f"""there is an upcoming match between {home}(home) and {away}(Away):

    based on the last 3 games played by {home}  and the last 3 games played by {away}, 
    provide average possession and shots on target for both team using the format below

            'team_home': ['{home}'],'team_away': ['{away}'],'possession_home': [0.47],'possession_away': [0.55],'shots_on_target_home': [3.67],'shots_on_target_away': [4.33]

        """
        print(p)
        return p

    def preprocess_data_old(self, data, label_encoder=None):
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

    def preprocess_data(self, data, label_encoder=None):
        # Convert percentage values to numeric
        def convert_possession(value):
            if isinstance(value, str) and '%' in value:
                return float(value.rstrip('%')) / 100.0
            else:
                return float(value) / 100

        data['possession_home'] = data['possession_home'].apply(convert_possession)
        data['possession_away'] = data['possession_away'].apply(convert_possession)

        # Encode team names using LabelEncoder
        if label_encoder is None:
            label_encoder = LabelEncoder()

        # Fit the encoder with the union of training and prediction data
        all_teams = pd.concat([data['team_home'], data['team_away']]).unique()
        label_encoder.fit(all_teams)

        data['team_home_encoded'] = label_encoder.transform(data['team_home'])
        data['team_away_encoded'] = label_encoder.transform(data['team_away'])

        return data, label_encoder

    def train_model(self, data):
        # Split the data into features (X) and target variables (y)

        X = data[
            ['team_home_encoded', 'team_away_encoded', 'possession_home', 'possession_away', 'shots_on_target_home',
             'shots_on_target_away']]
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

    def save_model(self, model, filename):
        joblib.dump(model, filename)

    def load_model(self, filename):
        return joblib.load(filename)

    def predict_scores(self, model_home, model_away, label_encoder, new_game):
        # Encode team names in the new game
        new_game['team_home_encoded'] = label_encoder.transform(new_game['team_home'])
        new_game['team_away_encoded'] = label_encoder.transform(new_game['team_away'])

        # Extract features for prediction
        X_new_game = new_game[
            ['team_home_encoded', 'team_away_encoded', 'possession_home', 'possession_away', 'shots_on_target_home',
             'shots_on_target_away']]

        # Predict scores for the new game
        predicted_goals_home = model_home.predict(X_new_game)
        predicted_goals_away = model_away.predict(X_new_game)

        return predicted_goals_home[0], predicted_goals_away[0]


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = JbaPredictWindow()
    window.setWindowFlag(Qt.WindowStaysOnTopHint)  # Make the window topmost
    window.show()
    sys.exit(app.exec_())

