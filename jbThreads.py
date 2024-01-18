from PyQt5 import QtCore
from promptGenerator import GeneratePrompt
'''
Usage:
  self.threadController[f"ImageLoader-{tempName}"] = LoadImageThread(self.myself, self.thumbnail)
        self.threadController[f"ImageLoader-{tempName}"].start()
        self.threadController[f"ImageLoader-{tempName}"].any_signal.connect(image_loader_connector)
'''


class GetPromptThread(QtCore.QThread):
    any_signal = QtCore.pyqtSignal(dict)

    def __init__(self, parent):
        super(GetPromptThread, self).__init__()
        self.mainWindow = parent
        self.data_to_emit = {}
        self.data_to_emit['error'] = ""
        self.data_to_emit['status'] = ''
        self.message = ""

    def stop(self):
        self.requestInterruption()

    def get_prompts(self):

        try:
            if self.mainWindow.auto_checkbox.isChecked():
                self.sport = self.mainWindow.get_selected_sport()
                # self.game_period = self.game_period_input.text()
                self.game_period = self.mainWindow.get_selected_content()
                self.prompt_generator = GeneratePrompt(sport=self.sport, period=self.game_period, mainWindow=self.mainWindow)
                self.prompt_generator.start(include_srl=self.mainWindow.include_srl.isChecked())
                self.data_to_emit['status'] = 'completed'
                self.any_signal.emit(self.data_to_emit)
                pass
            else:
                print("Single! Single!! Single!!!")
                self.sport = self.mainWindow.get_selected_sport()
                home_team = self.mainWindow.home_team_input_teams.text()
                away_team = self.mainWindow.away_team_input_teams.text()
                league = self.mainWindow.league_input.text()

                if home_team == "" or away_team == "" or league == "":
                    self.data_to_emit['error'] = "Data is required for Home team, Away team and league name field.\nYou can check the 'Auto' check box for multiple prompts"
                    self.any_signal.emit(self.data_to_emit)
                    raise Exception

                self.prompt_generator = GeneratePrompt(single=True, home=home_team, away=away_team, league=league, sport=self.sport)
                self.prompt_generator.start(include_srl=self.mainWindow.include_srl.isChecked())
                self.data_to_emit['status'] = 'Completed!'
                self.any_signal.emit(self.data_to_emit)

        except Exception as e:
            if str(e).__contains__('browser version'):
                required = str(e).split("Stacktrace")[0].split(" ")[-1]
                self.data_to_emit['error'] = f"Current chrome driver is outdated. Please download the current version to continue. \n\nRequired Version: {required}"
                self.any_signal.emit(self.data_to_emit)
            if str(e).lower().__contains__('net'):
                print(e)
                self.data_to_emit['error'] = "Operation Aborted, Possibly due to Internet Connection issue. Please check your Internet connection and try again"
                self.any_signal.emit(self.data_to_emit)
            else:
                if str(e) != "":
                    print(f"An Error occurred in get_prompt(): {e}")
                    self.data_to_emit['error'] = f"An Error occurred in get_prompt(): {e}"
                    self.any_signal.emit(self.data_to_emit)

    def run(self):
        try:
            if self.isInterruptionRequested() is True:
                raise Exception

            self.get_prompts()

        except Exception as e:
            print(f"An Error Occurred in LoadImageThread > run: {e}")
        pass


