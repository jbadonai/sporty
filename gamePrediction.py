'''
This was intended to be used to replace the working predictOutcome
'''

import sys
from PyQt5.QtWidgets import QMessageBox, QApplication, QWidget, QVBoxLayout, QGroupBox, QLabel, QLineEdit, QPushButton, \
    QTextEdit, QHBoxLayout, QVBoxLayout, QCheckBox, QComboBox, QRadioButton
from PyQt5.QtCore import Qt
from PyQt5.QtGui import  QPalette, QColor, QFont
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
import predictOutcome2, predictOutcome4, basketballPrediction, footbalPrediction
from prettytable import PrettyTable

from jbThreads import GetPromptThread


class JbaPredictWindow(QWidget):
    def __init__(self):
        super().__init__()

        # self.setWindowTitle("JBA Predict")
        self.setGeometry(100, 100, 800, 400)

        self.model_filename = 'football_pm.joblib'

        self.training_df = None
        self.training_data_filename = "football_team_training_data.xlsx"
        self.prompt_generator = None

        self.sport = "football"
        self.game_period = 6
        self.selected_algorithm = "Linear Regression"  # Default value

        self.dfDisplay = ""
        self.appTitle = "JBA Game Predict"
        self.setWindowTitle(self.appTitle)
        self.threadController = {}
        # self.auto_calculate_predict_data = True

        # i want to use data for both team only as data to predict
        # this counter helps to count just 2 data and reset the training data file once 2 training data is received
        self.single_game_counter = 0  # to be usd to ensure that only current game data is used for predicting

        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()

        # Top Section - Teams and Prompt Generation
        top_layout = QVBoxLayout()

        # GroupBox for Teams
        teams_groupbox = QGroupBox("Generate Prompts: (Required to get Training and Predict data from AI like Bing...")
        teams_layout = QHBoxLayout()

        # contents for teams groupbox and layout
        self.auto_checkbox = QCheckBox("Auto")
        self.auto_checkbox.stateChanged.connect(self.toggle_auto_mode)

        self.home_team_input_teams = QLineEdit()
        self.away_team_input_teams = QLineEdit()
        self.league_input = QLineEdit()
        self.game_period_input = QLineEdit()

        # Create combo box
        self.comboBox = QComboBox(self)
        self.comboBox.addItems(['1h', '3h', '6h', '24h', 'All'])

        self.home_team_input_teams.setObjectName("homeTeamInput")
        self.away_team_input_teams.setObjectName("awayTeamInput")
        self.league_input.setObjectName('leagueInput')

        # Connect the textChanged signal to the update_input_objects function
        self.home_team_input_teams.textChanged.connect(self.update_input_objects)
        self.away_team_input_teams.textChanged.connect(self.update_input_objects)
        self.league_input.textChanged.connect(self.update_input_objects)

        self.home_team_input_teams.setPlaceholderText("Home Team")
        self.away_team_input_teams.setPlaceholderText("Away Team")
        self.league_input.setPlaceholderText("League")
        # self.sport_input.setPlaceholderText("Sport")
        self.game_period_input.setPlaceholderText("Game Period")

        # self.sport_input.setText("football")
        self.game_period_input.setText("6")

        # Replace the existing self.sport_input lines with the following code
        self.sport_combobox = QComboBox(self)
        self.sport_combobox.addItems(['football', 'basketball','vfootball'])
        self.sport_combobox.setCurrentText('football')  # Set the default value

        self.get_prompts_button = QPushButton("Get Prompts")
        self.get_prompts_button.setObjectName("getPrompts")
        self.get_prompts_button.clicked.connect(self.get_prompts)

        # adding content to the layout
        teams_layout.addWidget(self.auto_checkbox)
        teams_layout.addWidget(self.home_team_input_teams)
        teams_layout.addWidget(self.away_team_input_teams)
        teams_layout.addWidget(self.league_input)
        teams_layout.addWidget(self.sport_combobox)
        teams_layout.addWidget(self.comboBox)
        teams_layout.addWidget(self.get_prompts_button)

        # adding layout to the groupbox
        teams_groupbox.setLayout(teams_layout)

        # create settings groupbox to be added to top layout
        team_settings_groupbox = QGroupBox("Settings")
        team_settings_layout = QHBoxLayout()

        # create settings items
        self.include_srl = QCheckBox("Include SRL   |")
        self.auto_predict_data = QCheckBox("Auto Data for Predict   |")
        self.recent_game_label = QLabel("Number of Recent Games:")
        self.recent_game_number = QComboBox(self)
        self.recent_game_number.addItems(["3", "4", "5", "7", "10"])
        self.recent_game_number.setCurrentText("5")

        # Create radio buttons for algorithm selection
        # self.linear_regression_radio = QRadioButton("Linear Regression")
        # self.random_forest_radio = QRadioButton("Random Forest")
        # Set up a signal to connect the radio buttons to a method
        # self.linear_regression_radio.setChecked(True)
        # self.linear_regression_radio.clicked.connect(lambda: self.set_selected_algorithm("Linear Regression"))
        # self.random_forest_radio.clicked.connect(lambda: self.set_selected_algorithm("Random Forest"))

        # add settings items to layout
        team_settings_layout.addWidget(self.include_srl)
        team_settings_layout.addWidget(self.auto_predict_data)
        team_settings_layout.addWidget(self.recent_game_label)
        team_settings_layout.addWidget(self.recent_game_number)
        # team_settings_layout.addWidget(self.linear_regression_radio)
        # team_settings_layout.addWidget(self.random_forest_radio)

        # add settings layout to settings groupbox
        team_settings_groupbox.setLayout(team_settings_layout)

        # adding groupbox to the top layout
        top_layout.addWidget(team_settings_groupbox)
        top_layout.addWidget(teams_groupbox)

        # add ing top layout to the main window
        main_layout.addLayout(top_layout)

        # used to arrange the button horizontally
        buttons_groupbox = QGroupBox("")
        buttons_layout = QHBoxLayout()

        # Bottom Section - Prediction
        prediction_layout = QVBoxLayout()

        prompt_code_label = QLabel("Training Data / Predict Data:")
        self.prompt_code_input = QTextEdit()
        # self.prompt_code_input.setFontPointSize(112)

        self.predict_button = QPushButton("Predict")
        # predict_button.clicked.connect(self.predict)
        self.predict_button.clicked.connect(self.start)

        self.load_train_data_button = QPushButton("Load Training Data")
        self.load_train_data_button.clicked.connect(self.load_training_data)

        prediction_result_label = QLabel("Prediction Result:")
        self.display_info = QLabel("Status")
        #-----------------------------------------------------
        # Set background color to black
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(0, 0, 0))  # RGB values for black
        self.display_info.setAutoFillBackground(True)
        self.display_info.setPalette(palette)

        # Set font color to yellow
        self.display_info.setStyleSheet("color: yellow;")

        # Enable word wrap
        self.display_info.setWordWrap(True)

        # Set text alignment to center
        self.display_info.setAlignment(Qt.AlignCenter)

        # Set font size to 8
        font = QFont()
        font.setPointSize(8)
        self.display_info.setFont(font)

        # Set text padding using style sheet
        padding = "padding: 10px;"  # Adjust the value (10px) based on your preference
        self.display_info.setStyleSheet(f"{padding} color: yellow;")
        #-----------------------------------------------------
        self.prediction_result_display = QTextEdit()
        self.prediction_result_display.setReadOnly(True)
        text_edit_font = QFont()
        text_edit_font.setPointSize(10)  # Set the font size to 14
        self.prompt_code_input.setFont(text_edit_font)
        self.prediction_result_display.setFont(text_edit_font)

        buttons_layout.addWidget(self.load_train_data_button)
        buttons_layout.addWidget(self.predict_button)

        buttons_groupbox.setLayout(buttons_layout)

        prediction_layout.addWidget(prompt_code_label)
        prediction_layout.addWidget(self.prompt_code_input)

        prediction_layout.addWidget(buttons_groupbox)
        prediction_layout.addWidget(prediction_result_label)
        prediction_layout.addWidget(self.prediction_result_display)
        prediction_layout.addWidget(self.display_info)

        self.prompt_code_input.textChanged.connect(self.clear_prediction)

        # Add Prediction section to the main layout
        main_layout.addLayout(prediction_layout)

        self.setLayout(main_layout)

    # Inside JbaPredictWindow class
    # def set_selected_algorithm(self, algorithm_name):
    #     self.selected_algorithm = algorithm_name
    #     # print(f"Selected Algorithm: {self.selected_algorithm}")

    def get_selected_sport(self, index=None):
        selected_sport = self.sport_combobox.currentText()
        return selected_sport

    def get_selected_content(self):
        selected_content = self.comboBox.currentText()

        # Map selected content to the desired output
        if selected_content == '1h':
            result = 1
        elif selected_content == '3h':
            result = 3
        elif selected_content == '6h':
            result = 6
        elif selected_content == '24h':
            result = 24
        elif selected_content == 'All':
            result = ''

        return result

    def update_input_objects(self):
        if self.home_team_input_teams.text() != "" or self.away_team_input_teams.text() != "" or self.league_input.text() != "":
            content = "not empty"
        else:
            content = ""

        # Enable or disable other objects based on whether the content is empty or not
        self.sport_combobox.setEnabled(not content)
        # self.game_period_input.setEnabled(not content)
        self.comboBox.setEnabled(not content)

    def toggle_auto_mode(self, state):
        if state == Qt.Checked:
            # Auto mode is checked, disable other components
            self.home_team_input_teams.clear()
            self.away_team_input_teams.clear()
            self.league_input.clear()
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
        def get_prompt_controller(data):
            if data['error'] != "":
                QMessageBox.critical(self, "Error!", data['error'])
                self.get_prompts_button.setEnabled(True)
            if data['status'] == "completed":
                self.get_prompts_button.setEnabled(True)
            if data['status'] != "":
                self.display_info.setText(data['status'])
            print(f"DATA:::{data}")
            pass
        recent_games = self.recent_game_number.currentText()
        print(f'recent game: {recent_games}')
        self.get_prompts_button.setEnabled(False)
        self.threadController[f"get_prompt"] = GetPromptThread(self, recent_games=recent_games)
        self.threadController[f"get_prompt"].start()
        self.threadController[f"get_prompt"].any_signal.connect(get_prompt_controller)

        # try:
        #     if self.auto_checkbox.isChecked():
        #         self.sport = self.get_selected_sport()
        #         # self.game_period = self.game_period_input.text()
        #         self.game_period = self.get_selected_content()
        #         self.prompt_generator = GeneratePrompt(sport=self.sport, period=self.game_period)
        #         self.prompt_generator.start(include_srl=self.include_srl.isChecked())
        #         pass
        #     else:
        #         print("Single! Single!! Single!!!")
        #         self.sport = self.get_selected_sport()
        #         home_team = self.home_team_input_teams.text()
        #         away_team = self.away_team_input_teams.text()
        #         league = self.league_input.text()
        #
        #         if home_team == "" or away_team == "" or league == "":
        #             QMessageBox.information(self, "Empty Data!", "Home team, Away team and league name is required!\n"
        #                                                          "otherwise, check the 'Auto' check box.")
        #             raise Exception
        #         self.prompt_generator = GeneratePrompt(single=True, home=home_team, away=away_team, league=league, sport=self.sport)
        #         self.prompt_generator.start(include_srl=self.include_srl.isChecked())
        #     # Add your logic here to use home_team and away_team to generate prompts
        #     # Update the UI or perform other actions as needed
        # except Exception as e:
        #     if str(e).__contains__('browser version'):
        #         required = str(e).split("Stacktrace")[0].split(" ")[-1]
        #         QMessageBox.information(self, "Outdated Chrome Driver",
        #                                 f"Current chrome driver is outdated. Please download the current version to continue. \n\nRequired Version: {required}")
        #     print(f"An Error occurred in get_prompt(): {e}")
        #     pass

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
                              possession_percentage, strength, head_to_head, opponent_is_home,
                              opponent_recent_performance,
                              opponent_average_goals, opponent_player_injuries, opponent_possession_percentage,
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

            input_data[['Team_Wins_recent', 'Team_Losses_recent', 'Team_Draws_recent']] = input_data[
                'Recent 5 Performance'].astype(
                str).str.split(':', expand=True).astype(int)
            input_data[['Team_Wins_head2head', 'Team_Losses_head2head', 'Team_Draws_head2head']] = input_data[
                'Head-to-Head'].astype(
                str).str.split(':', expand=True).astype(int)

            input_data[['opp_Wins_recent', 'opp_Losses_recent', 'opp_Draws_recent']] = input_data[
                'Opponent Recent 5 Performance'].astype(
                str).str.split(':', expand=True).astype(int)
            input_data[['opp_Wins_head2head', 'opp_Losses_head2head', 'opp_Draws_head2head']] = input_data[
                'Opponent Head-to-Head'].astype(
                str).str.split(':', expand=True).astype(int)

            # Drop unnecessary columns
            input_data = input_data.drop(['Recent 5 Performance', 'Head-to-Head', 'Opponent Recent 5 Performance',
                                          'Opponent Head-to-Head'], axis=1)

            # Convert 'Is Home' column to binary (1 for True, 0 for False)
            input_data['Is Home'] = input_data['Is Home'].astype(int)

            # Preprocess 'Possession Percentage' column
            input_data['Possession Percentage'] = input_data['Possession Percentage'].apply(
                self.preprocess_possession_percentage)
            input_data['Opponent Possession Percentage'] = input_data['Opponent Possession Percentage'].apply(
                self.preprocess_possession_percentage)

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
            df[['Home_Wins_recent', 'Home_Losses_recent', 'Home_Draws_recent']] = df[
                'Team Recent 5 Performance'].astype(str).str.split(':', expand=True).astype(int)
            df[['Home_Wins_head2head', 'Home_Losses_head2head', 'Home_Draws_head2head']] = df[
                'Team Head-to-Head'].astype(str).str.split(':', expand=True).astype(int)

            # Opponent
            print(2)
            df[['Opp_Wins_recent', 'Opp_Losses_recent', 'Opp_Draws_recent']] = df[
                'Opponent Recent 5 Performance'].astype(str).str.split(':', expand=True).astype(int)
            df[['Opp_Wins_head2head', 'Opp_Losses_head2head', 'Opp_Draws_head2head']] = df[
                'Opponent Head-to-Head'].astype(str).str.split(':', expand=True).astype(int)

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
        sport = self.get_selected_sport()
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
                                                home_player_injuries, home_possession_percentage, home_strenght,
                                                home_head_to_head,
                                                away_is_home, away_recent_5_performance, away_average_goals,
                                                away_player_injuries,
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

            if start is False and c == "[":
                start = True
                end = False

            if start is True and c == "]":
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
            prediction_list = []
            for data in data_list:
                try:
                    # print(data)
                    prompt_code = eval(data)

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

                    odd_diff = round(abs(h - a), 1)
                    secondOpinion = self.second_opinion(h, d, a, result)

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
        ph = h / total
        pd = d / total
        pa = a / total

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

        implied_probaility = (
        round((1 / home_odd) * 100, 2), round((1 / draw_odd) * 100, 2), round((1 / away_odd) * 100, 2))

        print(f"implied probaility: {implied_probaility}")

        odd_predict, odd_percentage = self.probaiblity(implied_probaility[0], implied_probaility[1],
                                                       implied_probaility[2])

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
        # check if the file exists. and exit if not
        print(home_team)
        print(away_team)
        if os.path.exists("leagueTable.txt") is False:
            return "Unknown - League Table not Found! Run Prompt Generator Again!"
        with open("leagueTable.txt", 'r') as f:
            data = f.read()

        league_table = eval(data)

        # if both home and away team are found: no missed spelt or incompelte name
        for home, away, league, clock in league_table:
            if str(home).lower().strip() == str(home_team).lower().strip() and str(away).lower().strip() == str(
                    away_team).lower().strip():
                return league, clock

        # if only home name is correct
        for home, away, league, clock in league_table:
            if str(away).lower().strip() == str(away_team).lower().strip():
                return league, clock, home_team, away_team

        # if only the away team name is correct
        for home, away, league, clock in league_table:
            if str(home).lower().strip() == str(home_team).lower().strip():
                return league, clock, home_team, away_team

        return "Unknown League Name not found in the League Table", "-.--"  # Return None if the match is not found

    def find_odds(self, home_team, away_team):
        if os.path.exists("odds.txt") is False:
            return "Unknown - odds Table not Found! Run Prompt Generator Again!"
        with open("odds.txt", 'r') as f:
            data = f.read()

        if data == "":
            return "Unknown - odds Table not Found! Run Prompt Generator Again!"

        odd_table = eval(data)

        for home, away, odds in odd_table:
            if str(home).lower().strip() == str(home_team).lower().strip() and str(away).lower().strip() == str(
                    away_team).lower().strip():
                return odds
        return [0, 0, 0]  # Return None if the match is not found

    # CODE FROM 'FOOTBALSCOREPREDICTION' BEGINS HERE
    # ======================================================
    def update_training_data_filename(self, sport):
        if sport == "football":
            self.training_data_filename = "football_team_training_data.xlsx"
        elif sport == "basketball":
            self.training_data_filename = "basketball_training_data.xlsx"
        pass

    def save_dataframe(self, df, sport='football'):
        try:

            self.update_training_data_filename(sport)
            print(f"Sport={sport}")
            print(f"data filename={self.training_data_filename}")

            if self.single_game_counter == 0:
                self.single_game_counter += 1
                raise FileNotFoundError
            # Try to read the existing data from the Excel file
            existing_df = pd.read_excel(self.training_data_filename, sheet_name='Sheet1')

            # Check if the new DataFrame is a subset of the existing DataFrame
            if df.isin(existing_df.to_dict(orient='list')).all().all():
                QMessageBox.information(self, "Data Already Existing!",
                                        "Data provided is already in the training data. Skipping duplicate data which might affect prediction")
                print('DataFrame is already a subset of the existing data. Skipping...')
            else:
                # self.dfDisplay = "[*][*]"
                # Concatenate the existing data with the new data
                combined_df = pd.concat([existing_df, df], ignore_index=True)

                # Write the combined DataFrame back to the Excel file
                combined_df.to_excel(self.training_data_filename, index=False, sheet_name='Sheet1')
                self.single_game_counter = 0
                print(f'[main]DataFrame has been appended to {self.training_data_filename}')

            self.setWindowTitle(f"{self.appTitle} <-> <->")
        except FileNotFoundError:
            # If the file doesn't exist, create a new Excel file with the DataFrame
            df.to_excel(self.training_data_filename, index=False, sheet_name="Sheet1")
            print(f'[error]New DataFrame has been written to {self.training_data_filename}')

            self.setWindowTitle(f"{self.appTitle} <->")

    def load_training_data(self):
        error_status = ""
        counter_limit = 0
        sport = "football"
        try:
            data = self.prompt_code_input.toPlainText().strip()
            if data.strip().lower() == "reset me":
                self.single_game_counter = 0
                raise Exception
            if data.lower().__contains__("field goal") or data.lower().__contains__("fg%") or data.lower().__contains__(
                    "quarterly"):
                counter_limit = 12
                sport = "basketball"
            else:
                counter_limit = 8
                sport = "football"

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
                    QMessageBox.information(self, "Wrong info!",
                                            "Wrong Training Data detected!\n Team name not found on the firs Line.")
                    self.prompt_code_input.clear()
                    raise Exception

                title = data.split("\n")[0].split("Last")[0]
                title = title.replace("-", "").replace(":", "").replace("'s", "").strip()
                # print(f"title: {title}")
            else:
                dataList = data.split("\n")
                dateData = False
                a = data.count("\n")
                if a < 3 or (data.split("\n")[0] != 'team_home' and data.split("\n")[0].lower() != 'team name' and
                             data.split("\n")[0].lower().__contains__('team name') is False):
                    QMessageBox.information(self, "Wrong info!", "Wrong Training Data detected!")
                    self.prompt_code_input.clear()
                    raise Exception

            first_letter_found = False  # to control spaces before table begins
            for d in dataList:
                if first_letter_found is False:
                    if d == "":
                        print('Empty found')
                        continue
                    else:
                        first_letter_found = True

                counter += 1
                line_data += d + ","
                if d.strip() == "-" or d.strip() == "?" or d.strip() == "TBD" or d.strip() == "" or \
                        d.strip() == "N/A" or d.strip().__contains__(":") or d.strip() == "CANC":
                    impurity = True

                if counter == counter_limit:
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

            if sport == "football":
                header = ["team_home", "team_away", "goals_home", "goals_away", "possession_home", "possession_away",
                          "shots_on_target_home", "shots_on_target_away"]
            elif sport == "basketball":
                header = ['Team name', 'Team FG%', 'Team FT%', 'Team 3P%', 'Team quarterly scores', 'Team total points',
                          'Opponent name', 'Opponent FG%', 'Opponent FT%', 'Opponent 3P%', 'Opponent quarterly scores',
                          'Opponent total points']

            # print(df.columns)
            df.columns = header

            if dateData is True:
                df['team_home'] = title

            # update the training data filename base on the sport
            self.update_training_data_filename(sport)

            # save the training data to file
            self.save_dataframe(df, sport)
            self.prompt_code_input.clear()

        except Exception as e:
            self.prompt_code_input.clear()
            if error_status != "":
                pass

            if str(e) != "":
                print(f" An error occurred in load training data: {e}")
            else:
                pass
        pass

    def re_train(self):
        try:
            print(self.training_data_filename)
            if os.path.exists(self.training_data_filename) is False:
                print("file not found!")
                raise FileNotFoundError
            print(1)
            data = pd.read_excel(self.training_data_filename)
            print(2)
            print(data)

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

            return label_encoder
        except FileNotFoundError as e:
            return "File Not Found!"
        except Exception as e:
            return f"Unknown Error! {e}"
            pass

    def winner_by_odd(self, h, d, a):
        # this function check  which team with the lowest odd

        winner = None
        numbers = [h, d, a]

        # Find the minimum among the three numbers
        smallest = min(numbers)

        # Find the position of the smallest number
        position = numbers.index(smallest)

        if position == 0:
            winner = "Home"
        if position == 1:
            winner = "Draw"
        if position == 2:
            winner = "Away"

        return winner

        pass

    def winner_by_prediction(self, h, a):
        # this function check  which team with the lowest odd
        print(f"winer by prediction input home: {h} - away {a}")
        winner = None

        numbers = [h, a]

        # Find the minimum among the three numbers
        smallest = max(numbers)

        # Find the position of the smallest number
        position = numbers.index(smallest)
        if h == a:
            winner = "Draw"
        elif position == 0:
            winner = "Home"
        elif position == 1:
            winner = "Away"

        print(f"winner is {winner} at position {position}")
        return winner

    import pandas as pd

    def calculate_average(self, target_number):
        # Load the Excel file into a pandas DataFrame
        df = pd.read_excel('football_team_training_data.xlsx')

        # Get the unique teams in the data
        teams = df['team_home'].unique()
        print(teams)

        # Initialize dictionaries to store the last 'target_number' records for each team
        last_records = {col: [] for col in df.columns}

        # Iterate over each team and extract the last 'target_number' records
        for team in teams:
            team_data = df[(df['team_home'] == team) | (df['team_away'] == team)].tail(target_number)
            for col in df.columns:
                last_records[col].extend(team_data[col].tolist())

        print(last_records)

        # Calculate the average for the required columns
        average_data = {
            'team_home': [last_records['team_home'][0]],
            'team_away': [last_records['team_away'][0]],
            'possession_home': [sum(float(val.strip('%')) for val in last_records['possession_home']) / (target_number*2)],
            'possession_away': [sum(float(val.strip('%')) for val in last_records['possession_away']) / (target_number*2)],
            'shots_on_target_home': [sum(last_records['shots_on_target_home']) / (target_number*2)],
            'shots_on_target_away': [sum(last_records['shots_on_target_away']) / (target_number*2)]
        }

        return average_data

    def start(self):
        try:
            # Get prompt input from the text box
            predict_prompt = self.prompt_code_input.toPlainText()

            if self.auto_predict_data.isChecked() is False:
                # check if input is not empty
                if predict_prompt == "":
                    QMessageBox.information(self, "No Team Info", "Team info to predict are required! Please "
                                                                  "provide team info in the required format")
                    raise Exception

                # validating the input prompt by counting the number of ','
                a = predict_prompt.count(",")
                if a != 5 and a != 7:
                    QMessageBox.information(self, "Wrong info!", "Wrong Predict prompt detected!")
                    self.prompt_code_input.clear()
                    raise Exception

                print(f"a={a}")

                sport = None
                if a == 5:
                    sport = "football"
                elif a == 7:
                    sport = "basketball"

                # put the prompt in a new container for data cleaning
                new_game_data = predict_prompt

                # scan through every character in the prompt to remove unlikely characters and replace them with right ones
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

                # check if prompt input string looks like dictionary if not add curl bracket
                if updated.__contains__("{") is False:
                    updated = "{" + updated + "}"

                # convert the string into dictionary
                new_game_data = eval(updated)
            else:
                # Example usage with target_number = 2
                auto_data = self.calculate_average(3)
                a = str(auto_data).count(",")

                print(f"autodata: a={a}")

                sport = None
                if a == 5:
                    sport = "football"
                elif a == 7:
                    sport = "basketball"

            if sport == "basketball":
                basketball_prediction_output, bb_home_team_name, bb_away_team_name = self.start_basketball_prediction(
                    new_game_data)
                prediction_dict = basketball_prediction_output[0]
                bb_home_score = prediction_dict['team_points']
                bb_away_score = prediction_dict['opponent_points']
                bb_home_quaters = prediction_dict['team_quarters']
                bb_away_quaters = prediction_dict['opponent_quarters']

                bb_data = basketball_prediction_output[1]
                # print(bb_data.columns)

                homeQ = bb_home_quaters.split(" - ")
                awayQ = bb_away_quaters.split(" - ")

                bb_home_team_name = bb_home_team_name[0]
                bb_away_team_name = bb_away_team_name[0]

                league, clock = self.find_league(bb_home_team_name, bb_away_team_name)
                league = league.split(' ')[1:]
                league = " ".join(league)

                bb_result = f"""
                [{clock}]- {league}
                {'-' * (len(clock) + len(league)+ 5) }
                {bb_home_team_name}(Home)    {round(float(bb_home_score))} : {round(float(bb_away_score))}    {bb_away_team_name}(Away) 
                TOTAL SCORES: {round(float(bb_home_score) + float(bb_away_score))}

                QUARTERLY SCORES:
                Home:- [Q1:{round(float(homeQ[0]))}] - [Q2:{round(float(homeQ[1]))}] - [Q3:{round(float(homeQ[2]))}] - [Q4:{round(float(homeQ[3]))}]
                Away:- [Q1:{round(float(awayQ[0]))}] - [Q2:{round(float(awayQ[1]))}] - [Q3:{round(float(awayQ[2]))}] - [Q4:{round(float(awayQ[3]))}]
                """

                self.prompt_code_input.clear()
                self.prediction_result_display.setText(bb_result)
                self.write_prediction('predictions_bb.txt', bb_result)

            elif sport == "football":
                if self.auto_predict_data.isChecked() is False:
                    football_prediction_outcome = self.start_football_prediction(new_game_data)
                else:
                    football_prediction_outcome = self.start_football_prediction(auto_data)

                football_home_team = football_prediction_outcome['home_team_name']
                football_away_team = football_prediction_outcome['away_team_name']
                football_home_goal = football_prediction_outcome['predicted_goals_home']
                football_away_goal = football_prediction_outcome['predicted_goals_away']
                football_algorithm_used = football_prediction_outcome['algorithm_used']

                ans = self.find_league(football_home_team, football_away_team)
                if len(ans)==2:
                    league, clock = ans
                elif len(ans) == 4:
                    league, clock, ht, at = ans

                league = league.split(' ')[1:]
                league = " ".join(league)
                football_result = f"""
                    [{clock}]- {league}
                    {football_home_team}(Home)   {round(football_home_goal)} : {round(football_away_goal)}   {football_away_team}(Away)
                    
                    {football_home_team}   {round(football_home_goal,3)} : {round(football_away_goal,3)}   {football_away_team}
                    """
                self.prompt_code_input.clear()
                self.prediction_result_display.setText(football_result)
                self.write_prediction('predictions_fb.txt', football_result)
                pass

            self.setWindowTitle(f"{self.appTitle} - Ready!")
            pass
        except FileNotFoundError as e:
            QMessageBox.information(self, "No Training Data", "Training Data required! Load Training data!")
            self.prompt_code_input.clear()

        except Exception as e:
            self.prediction_result_display.setText(str(e))
            if str(e).__contains__('unseen labels'):
                t = str(e).split(":")[-1]
                QMessageBox.information(self, "Missing Data!",
                                        f"Missing data for  team {t} in the training data. Previous 10 game statistics for {t} is required in the training data, for successful prediction. \n"
                                        f"Or Check that the team name is not abrreviated or shortened in the training data. \n"
                                        f"This error can also occurr if for example you have 'Abu Salim SC' in training data but you are tryinig to predict score for 'Abu Salim' without the 'SC'. ")
            print(f"An Error occurred in start: {e}")

    def start_football_prediction(self, rawData):
        # Instantiate the PredictOutcome class
        print('initializing football training class...')
        predictor = footbalPrediction.PredictOutcome()

        # Train the models and select the best algorithm based on MSE
        print("Training football prediction model...")
        predictor.train_model()

        # Load the trained model
        print("Loading Trained model...")
        predictor.load_model()

        # Sample data for prediction
        print('Getting input data for prediction...')
        input_data = rawData

        # Make prediction
        print("making prediction...")
        result = predictor.predict_scores(input_data)
        return result

        pass

    def train_basketball(self):
        # Initialize the PredictOutcome class
        predictor = basketballPrediction.PredictOutcome(algorithm=self.selected_algorithm)

        # Load and preprocess the data
        data = predictor.load_data("basketball_training_data.xlsx")
        processed_data = predictor.preprocess_data(data)

        # Train the model and save it to disk
        print("Training models....")
        trained_model = predictor.train_model(processed_data)
        print("Saving model....")
        predictor.save_model(trained_model, "trained_model.joblib")

        return predictor
        pass

    def start_basketball_prediction(self, rawData):

        predictor = self.train_basketball()

        # Load the trained model from disk
        print("Loading trained model...")
        predictor.load_model("trained_model.joblib")
        print("model loaded")

        # Prepare input data for prediction
        # input_data = {
        #     'team_home': ['Kinlung Pegasus'],
        #     'team_away': ['South China'],
        #     'field_goal_percentage_home': [34.1],
        #     'free_throw_percentage_home': [70.0],
        #     'three_point_percentage_home': [26.8],
        #     'field_goal_percentage_away': [48.4],
        #     'free_throw_percentage_away': [78.5],
        #     'three_point_percentage_away': [35.1]
        # }
        input_data = rawData
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
        home_team_name = input_data['team_home']
        away_team_name = input_data['team_away']

        reordered_input_data['Team'] = input_data['team_home']
        reordered_input_data['Opponent'] = input_data['team_away']

        # Create a DataFrame from reordered input data
        input_df = pd.DataFrame(reordered_input_data)

        # print(input_df)
        # input("wait...")

        # Make predictions
        print("predicting.....")
        predictions = predictor.predict(input_df)
        return predictions, home_team_name, away_team_name

        pass

    def decide(self, home_score, away_score):
        diff = abs(abs(home_score) - abs(away_score))
        highest = None
        if abs(home_score) > abs(away_score):
            highest = "Home"
        else:
            highest = 'Away'

        if 0 < diff <= 1:
            if highest == "Home":
                return f"{highest} or Draw"
            else:
                return f"Draw or {highest}"

        elif 1 < diff <= 2:
            return "Home or Away"

        elif diff > 2:
            return highest
        else:
            return "Can't Decide"

    def decide_new(self, v2, v4, odd):
        """
            * !if v2 and v4 and odds alligns >>[SINGLE] odd (either direct home/away)
            * !if v2 and v4 alligns but not odds >> [DOUBLE] odds or v-value
            * !if v2 and v4 not align, v4 aligns with odd >> [SINGLE] odd
            * !if v2 and v4 not align, v2 aligns with odd >> [DOUBLE] v4 value or odd
            * if v2 and v4 not align, none align with odd >> {DOUBLE} trying... odd or draw
        """
        final_value = None
        print('---------------------------')
        print(f"{v2} -- {v4} -- {odd}")
        print('---------------------------')

        if v2 == v4 == odd:
            final_value = odd
        elif v2 == v4 and v4 != odd:
            final_value = f"{odd} or {v2}"
        elif v2 != v4 and v4 == odd:
            final_value = odd
        elif v2 != v4 and v2 == odd:
            final_value = f"{v4} or {odd}"
        elif v2 != v4 and v2 != odd and v4 != odd:
            final_value = f"{odd} or {v4}"
        else:
            final_value = "Can't decide"

        print(f"Final value = {final_value}")
        return final_value

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
        try:
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
        except Exception as e:
            print(f"An error occurred in Preprocess data: {e}")
            QMessageBox.information(self, "Error in Data File!",
                                    f"There is a data error in '{self.training_data_filename}'. Data Error could not be automatically resolved. Please inspect the file for inconsistent data and fix the error manually! Detail below could assist you. \n\nError Details:\n{e}")

    def train_model(self, data):
        # Split the data into features (X) and target variables (y)
        print(1)
        X = data[
            ['team_home_encoded', 'team_away_encoded', 'possession_home', 'possession_away', 'shots_on_target_home',
             'shots_on_target_away']]
        y_home = data['goals_home']
        y_away = data['goals_away']
        print(2)
        # Train a linear regression model for home goals
        model_home = LinearRegression()
        model_home.fit(X, y_home)
        print(3)
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

