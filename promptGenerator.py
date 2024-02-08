# try:
import subprocess
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
from datetime import datetime
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
import os
import requests
import pandas as pd
import time

def get_team_strength_prompt(home, away):
    teamStrengthPrompt = f'''
    
    provide the team strength of the 2 football teams '{home}' and '{away}' using the team statistics below based on the latest data:
    
    
        'win_loss_ratio':       # Win-loss ratio (float between 0 and 1)
        'goals_for'               # Average goals scored per game
        'goals_against'            # Average goals conceded per game
        'possession_percentage'     # Possession percentage (integer)
        'pass_completion_rate'      # Pass completion rate (percentage)
        'clean_sheets_ratio':       # Clean sheets ratio (float between 0 and 1)
    '''
    return teamStrengthPrompt

'''
 Team name: Mannarino, Adrian
League: ATP Australian Open Men Singles


            in a tabular form, provide the last 5 tennis games statistics  played  by Mannarino, Adrian,
            using the following header
            
                player name 
                opponent name
                player Aces
                opponent Aces
                player first serve percentage
		opponent's first serve percentage
		player's service game won
		opponent's service game won
		player's break points converted
		opponent break points converted
		player's first serve return points won
		opponent first server return points won
		player return games won
		opponent return game won
		player sets scores (set1-set2-set3-set4)
		opponent sets scores (set1-set2-set3-set4)
		player wins(True/False)
		opponent wins(True/false)
'''


class GeneratePrompt():
    def __init__(self, sport="football", period=6, single=False, home=None, away=None, league=None, mainWindow=None, recentGameNo=5):
        self.sport = sport
        self.start_time = period
        self.single = single
        self.single_home = home
        self.single_away = away
        self.single_league = league
        self.mainWndow = mainWindow
        self.recent_game_numbers = recentGameNo
        # self.prompt_style = 1
        if self.start_time != "":
            self.url = f"https://www.sportybet.com/ng/sport/{self.sport}?time={self.start_time}"
        else:
            self.url = f"https://www.sportybet.com/ng/sport/{self.sport}"

        self.first_level_target_class = "match-league-wrap"
        self.all_games = []
        self.league_table = []

    def get_scrape_prompt(self,sport,country, league):

        scrape_prompt = f"""
        for all the {sport} team in {country}, in {league} league,:

        get the following data in tabular form with reference to the last match played (No explanation required):

        team_name (team name),
        team_is_home(True if home),
        team_recent {self.recent_game_numbers} performance (team, in format w:l:d eg 2:0:1,  Don't use '-' delimeter),
        team_average_goals (team),
        team_player_injuries( team, use 0 if none),
        team_possession_percentage team),
        team_strenght (team, Elo rating actual not in percentage)
        head_to_head (team, in format w:l:d eg 2:0:1, Don't use '-' delimeter),
        opponent_team_name (opponent team),
        opponent_is_home(True if opponent),
        opponent_recent {self.recent_game_numbers} performance (opponent team, in format w:l:d eg 2:0:1,  Don't use '-' delimeter),
        opponent_average_goals (opponent team),
        opponent_player_injuries(opponent team, use 0 if none),
        opponent_possession_percentage (opponent team),
        opponent_strenght (opponent, Elo rating actual not in percentage)
        head_to_head (opponent, in format w:l:d eg 2:0:1, Don't use '-' delimeter),
        match outcome (Home win, Away win or Draw)

        """


        return scrape_prompt

    def get_header_prompt(self, sport):
        # prompt_header = f"in an upcoming {sport} match, between the following teams:\n"
        prompt_header = ""

        return prompt_header

    def get_body_prompt(self, league, home, away):
        # prompt_body = f"{home} (home) vs {away} (away) in [ {league} league ]:"
        prompt_body = f"Team Name: {home} \nLeague: {league} -- Team name: {away} \nLeague:{league}\n"

        return prompt_body

    def get_predict_prompt(self,home, away, league, sport="football"):
        p = f"""in an upcoming {sport} match between {home}(home) and {away}(Away), in  {league} league:
    using data from www.sofascore.com only,
    * get the last 3 games statistics for {self.get_predict_stat(sport)} data by {home}  and 
    * get the last 3 games statistics for {self.get_predict_stat(sport)} data by {away}, 
    * provide average {self.get_predict_stat(sport)} for both team using the json format below

            {self.get_json(sport, home, away)}
        """
        return p

    def get_predict_stat(self, sport):
        stat = ""
        if str(sport).lower() == 'football' or str(sport).lower() == 'vfootball':
            stat = "possession and shots on target"
        if str(sport).lower() == 'basketball':
            stat = "field goal percentage, free throw percentage and three-point percentage"

        return stat

    def get_json(self,sport, home, away):
        json = None
        if str(sport).lower() == 'football' or str(sport).lower() == 'vfootball':
            json = f"'team_home': ['{home}'],'team_away': ['{away}'],'possession_home': [0],'possession_away': [0],'shots_on_target_home': [0],'shots_on_target_away': [0]"
        elif str(sport).lower() == 'basketball':
            json = f"'team_home': ['{home}'],'team_away': ['{away}']," \
                f"'field_goal_percentage_home': [0]," \
                f"'free_throw_percentage_home': [0]," \
                f"'three_point_percentage_home': [0]," \
                f"'field_goal_percentage_away': [0]," \
                f"'free_throw_percentage_away': [0]," \
                f"'three_point_percentage_away': [0]"

        return json
        pass


#
#     def get_action_prompt_old(self, league="", home ="", away=""):
#         prompt_action = f"""
# get the following data (No explanation requried):
#
# home_team_name (home team),
# home_is_home(True if home),
# home_recent {self.recent_game_numbers} performance (home team, in format w:l:d eg 2:0:1,  Don't use '-' delimeter),
# home_average_goals (home team),
# home_player_injuries(home team, use 0 if none),
# home_possession_percentage (home team),
# home_strenght (home, Elo rating actual not in percentage)
# head_to_head (home, in format w:l:d eg 2:0:1, Don't use '-' delimeter),
# away_team_name (away team),
# away_is_home(True if away),
# away_recent {self.recent_game_numbers} performance (away team, in format w:l:d eg 2:0:1,  Don't use '-' delimeter),
# away_average_goals (away team),
# away_player_injuries(away team, use 0 if none),
# away_possession_percentage (away team),
# away_strenght (away, Elo rating actual not in percentage)
# head_to_head (away, in format w:l:d eg 2:0:1, Don't use '-' delimeter),
#
# Present the resulting 16 feature data of each team in a list of python turple. using strictly, the format below:
#
#        [ (#home_team_name#, #APO Levadiakos FC#),
#         (#home_is_home#, True),
#         (#home_recent_{self.recent_game_numbers}_performance#,#2:1:2#),
#         (#home_average_goals#, 1.4),
#         (#home_player_injuries#, 0),
#         (#home_possession_percentage#, 48),
#         (#home_strength#, 1310),
#         (#home_head_to_head#, #1:1:1#),
#         (#away_team_name#, #PAS Lamia 1964#),
#         (#away_is_home#, False),
#         (#away_recent_{self.recent_game_numbers}_performance#, #1:2:2#),
#         (#away_average_goals#, 0.8),
#         (#away_player_injuries#, 1),
#         (#away_possession_percentage#, 46),
#         (#away_strength#, 1291),
#         (#away_head_to_head#,#1:1:1#) ]
# """
#
#         return prompt_action

    def get_action_prompt(self, league="", home ="", away="",game="football", focus=""):
        if focus == "":
            prompt_action = f"""

in a tabular form, using data from www.sofascore.com only, provide the last {self.recent_game_numbers} games statistics  played  the team above,
using the following header:
team name 
opponent name
goal scored by team
goal scored by opponent
possession percentage by team
possession percentage by opponent
shots on target by team
shots on target by opponent
"""

        else:
            prompt_action = f"""
            in a tabular form, using data from www.sofascore.com, provide the last {self.recent_game_numbers} {game} games statistics  played  by {focus},
            using the following header
            {self.get_statistics_headers(game)}
            """

        return prompt_action

    def get_statistics_headers(self, game):
        stat = ""
        if str(game).lower() == "football" or str(game).lower() == "vfootball":
            stat = """
                team name 
                opponent name
                goal scored by team
                goal scored by opponent
                possession percentage by team
                possession percentage by opponent
                shots on target by team
                shots on target by opponent
                            """
        elif str(game).lower() == "basketball":
            stat = """
                    team name 	   
                    team field goal percentage
                    team free throw percentage
                    team three-point percentage
                    team quarterly scores
                    team total points	  
                    opponent name   
                    opponent field goal percentage
                    opponent free throw percentage
                    opponent three-point percentage
                    opponent quarterly scores
                    opponent total points
                    """
        return stat

    def open_file(self, filename):
        try:
            subprocess.Popen(["notepad.exe", filename])
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            self.mainWndow.display_info.setText(f"An error occurred: {str(e)}")

    def get_league_wrap(self, content):
        soup = BeautifulSoup(content, 'html.parser')
        all_leagues = soup.find_all(class_="match-league-wrap")
        return all_leagues

    # Define a custom sorting key function
    def custom_sort_key(self, item):
        # Parse the time string into a datetime object
        time_str = item[0]
        time_obj = datetime.strptime(time_str, "%H:%M")
        return time_obj

    def extract_odds(self, content):
        # consist of 2 markets left(1,x,2) and right(goals,over,under) each of them with
        # class name = m-market market
        market_data = content.select('.m-market.market')

        # select the first one which is for home/draw/away
        market_home_or_away = market_data[0]

        # extract the odd
        odds_elements = market_home_or_away('span', class_='m-outcome-odds')
        odds = [element.text for element in odds_elements]
        return odds

    def load_url(self, url):
        self.all_games.clear()

        print("loading Chrome...")
        self.mainWndow.display_info.setText("Loading Chrome...")
        chrome_options = Options()
        chrome_options.add_argument("--headless")  # Run Chrome in headless mode (no GUI)
        chrome_options.add_experimental_option('excludeSwitches', ['enable-logging'])

        driver = webdriver.Chrome(options=chrome_options)  # Make sure chromedriver is in your PATH

        print(f"loading url [{url}]...")
        self.mainWndow.display_info.setText(f"loading url [{url}]...")
        driver.get(url)  # Load the page once
        time.sleep(2)
        content = driver.page_source

        print('Extracting Content...')
        self.mainWndow.display_info.setText('Extracting Content...')
        leagues = self.get_league_wrap(content)

        for league in leagues:
            span = league.find(class_="league-title")
            # lig = span.get_text(strip=True)
            lig = span.text
            teams = []

            home_teams = league.find_all(class_='home-team')
            away_teams = league.find_all(class_='away-team')
            clock_time = league.find_all(class_='clock-time')
            # markets = league.find_all(class_='m-table-cell market-cell two-markets')
            markets = league.select('.m-table-cell.market-cell.two-markets')

            for index, x in enumerate(range(len(home_teams))):
                home = home_teams[index].text
                away = away_teams[index].text
                clock = str(clock_time[index].text).strip()
                odds = self.extract_odds(markets[index])

                g = {'home': home, 'away': away, 'clock': clock, 'odds': odds}

                teams.append(g)

            game = {'league': lig, 'teams': teams}
            self.all_games.append(game)

    def scrape(self):
        print("loading Chrome...")
        self.mainWndow.display_info.setText("loading Chrome...")
        chrome_options = Options()
        chrome_options.add_argument("--headless")  # Run Chrome in headless mode (no GUI)
        chrome_options.add_experimental_option('excludeSwitches', ['enable-logging'])

        driver = webdriver.Chrome(options=chrome_options)  # Make sure chromedriver is in your PATH

        url = "https://www.sportybet.com/ng/sport/football/"
        driver.get(url)  # Load the page once
        content = driver.page_source

        # Parse the html content using BeautifulSoup
        soup = BeautifulSoup(content, "html.parser")

        # Find all the div elements that contain the league names

        category_list_item = soup.find_all("li", class_="category-list-item")

        # Create an empty list to store the league names
        leagues = []
        categories = []

        with open('League_prompt.txt', 'w', encoding='utf-8') as f:
            f.write("")

        with open('scrape_prompt_new.txt', 'w', encoding='utf-8') as f:
            f.write("")

        counter_check = 0

        for list_item in category_list_item:
            counter_check += 1
            if counter_check <= 7:
                continue
            category_item = list_item.find("div", class_="category-item")
            title = category_item.find("span").text

            divs = list_item.find_all("li", class_="tournament-list-item")

            # Loop through the div elements and extract the league names
            for div in divs:
                # Get the text content of the div element
                text = div.get_text()
                # Strip any whitespace characters
                text = str(text.strip()).split("(")[0]

                league_prompt = get_scrape_prompt("football", title, text)

                with open('scrape_prompt_new.txt', 'a', encoding='utf-8') as f:
                    f.write(f"{'*' * 60}\n")
                    f.write(f"{title}\n")
                    f.write(f"{'*' * 60}\n")
                    f.write(league_prompt)
                # Append the text to the leagues list
                leagues.append(text)
                categories.append(title)

        data = {"Category": categories, "League": leagues}

        # Create a pandas dataframe from the leagues list
        df = pd.DataFrame(data)

        # df = pd.DataFrame(leagues, columns=["League"])

        # Save the dataframe to an excel file
        df.to_excel("leagues.xlsx", index=False)

        # Print a message to indicate the task is done
        print("The excel file with the league names has been saved.")
        self.mainWndow.display_info.setText("The excel file with the league names has been saved.")

    def start_single(self, home, away, league, sport="football"):
        self.clean_files()
        # prompt_header = self.get_header_prompt(self.sport)
        prompt_body = self.get_body_prompt(league, home, away)
        prompt_action_h = self.get_action_prompt(focus=home, game=sport)
        prompt_action_a = self.get_action_prompt(focus=away, game=sport)

        bp = prompt_body.split("--")

        # self.write_header('generated_prompt.txt', prompt_header)
        self.write_body('generated_prompt.txt', bp[0])
        self.write_action('generated_prompt.txt', prompt_action_h)
        self.write_body('generated_prompt.txt', bp[1])
        self.write_action('generated_prompt.txt', prompt_action_a)

        pdp = self.get_predict_prompt(home, away, league, sport=sport)
        self.write_predict_data_prompt('generated_prompt.txt', pdp)

        data = (home, away, league, "N/A")

        # load the existing data in leageu table if any
        existing_table = ""
        if os.path.exists("leagueTable.txt") is True:
            with open("leagueTable.txt", 'r', encoding='utf-8', errors='replace') as f:
                existing_table = f.read()
        else:
            existing_table = ""


        # append this new one to the existing table if there is any
        if existing_table != "":
            existing_table = eval(existing_table)
            existing_table.append(data)
            with open("leagueTable.txt", 'w', encoding='utf-8', errors='replace') as f:
                f.write(str(existing_table))
        else:
            # write a new file. overwrite the file with new single data
            with open("leagueTable.txt", 'w', encoding='utf-8', errors='replace') as f:
                f.write(f"[{data}]")

        self.open_file('generated_prompt.txt')
        pass

    def clean_files(self):
        with open('generated_prompt.txt', 'w', encoding='utf-8') as f:
            f.write("")

        with open('sorted_generated_prompt.txt', 'w', encoding='utf-8') as f:
            f.write("")

        with open('odds.txt', 'w', encoding='utf-8') as f:
            f.write("")

        with open('predict_data_prompt.txt', 'w', encoding='utf-8') as f:
            f.write("")

    def write_header(self, filename, prompt_header):
        with open(filename, 'a', encoding='utf-8', errors='replace') as f:
            f.write(prompt_header)
            f.write("\n")

    def write_body(self, filename, prompt_body):
        with open(filename, 'a', encoding='utf-8', errors='replace') as f:
            try:
                f.write(f"{prompt_body}\n\n")

            except:
                f.write(f"Skipped due to error\n")
                pass

    def write_action(self, filename, prompt_action):
        try:
            with open(filename, 'a', encoding='utf-8', errors='replace') as f:
                f.write(prompt_action)
                f.write("\n")
                f.write('-' * 80)
                f.write("\n")
        except Exception as e:
            print("error in write action..............")
            self.mainWndow.display_info.setText("error in write action..............")

    def write_predict_data_prompt(self, filename, prompt_body):
        with open(filename, 'a', encoding='utf-8', errors='replace') as f:
            try:
                f.write(f"{prompt_body}\n")
                f.write(f"{'^' * 80}\n\n")
                f.write(f"{'^' * 80}\n\n")
            except:
                f.write(f"* Skipped due to error\n")
                pass

    def start(self, include_srl=False):
        if self.single is False:
            sport = self.sport
            print(f"Sport! sport!! = {sport}")
            start_time = self.start_time
            if start_time != "":
                url = f"https://www.sportybet.com/ng/sport/{sport}?time={start_time}"
            else:
                url = f"https://www.sportybet.com/ng/sport/{sport}"

            # gets all games
            self.load_url(url)

            self.clean_files()

            data = []
            print('[S] Writing extracted content to file...')
            self.mainWndow.display_info.setText('[S] Writing extracted content to file...')

            def write_header(filename, prompt_header):
                with open(filename, 'a', encoding='utf-8', errors='replace') as f:
                    f.write(prompt_header)
                    f.write("\n")

            def write_body(filename, prompt_body):
                with open(filename, 'a', encoding='utf-8', errors='replace') as f:
                    try:
                        f.write(f"{prompt_body}\n\n")

                    except:
                        f.write(f"Skipped due to error\n")
                        pass

            def write_predict_data_prompt(filename, prompt_body):
                with open(filename, 'a', encoding='utf-8', errors='replace') as f:
                    try:
                        f.write(f"{prompt_body}\n")
                        f.write(f"{'^' * 80}\n\n")
                        f.write(f"{'^' * 80}\n\n")
                    except:
                        f.write(f"* Skipped due to error\n")
                        pass

            def write_action(filename, prompt_action):
                try:
                    with open(filename, 'a', encoding='utf-8', errors='replace') as f:
                        f.write(prompt_action)
                        f.write("\n")
                        f.write('-' * 80)
                        f.write("\n")
                except Exception as e:
                    print("error in write action..............")
                    self.mainWndow.display_info.setText("error in write action..............")

            def write_odds( home:str, away:str, odds:list):
                try:
                    data = f"{home},{away},{odds[0]},{odds[1]},{odds[2]}\n"
                    with open('odds.txt', 'a', encoding='utf-8', errors='replace') as f:
                        f.write(data)
                except Exception as e:
                    print("error in write action..............")
                    self.mainWndow.display_info.setText("error in write action..............")

            header_written = False
            all_odds =[]
            for game in self.all_games:
                for team in game['teams']:
                    league = game['league']
                    home = team['home']
                    away = team['away']
                    clock = team['clock']
                    odds = team['odds']

                    # write_odds(home, away, odds)
                    all_odds.append((home,away,odds))

                    self.league_table.append((home, away, league, clock))

                    prompt_header = self.get_header_prompt(sport)
                    prompt_body = self.get_body_prompt(league,home, away)

                    # write the header once
                    if header_written is False:
                        write_header('generated_prompt.txt', prompt_header)
                        header_written = True

                    # write the body
                    try:
                        write_body('generated_prompt.txt', prompt_body)
                    except:
                        pass

                    data.append((clock, prompt_header, prompt_body, league, home, away))

            # save the league table
            print("saving league table.............")
            self.mainWndow.display_info.setText("saving league table.............")
            with open("leagueTable.txt", 'w', encoding='utf-8', errors='replace') as f:
                f.write(str(self.league_table))

            print("saving odds table.............")
            self.mainWndow.display_info.setText("saving odds table.............")
            with open("odds.txt", 'w', encoding='utf-8', errors='replace') as f:
                f.write(str(all_odds))

            prompt_action = self.get_action_prompt(game=sport)
            write_action('generated_prompt.txt', prompt_action)

            print('sorting data....')
            self.mainWndow.display_info.setText('sorting data....')
            # Sort the list based on the custom sorting key
            sorted_data = sorted(data, key=self.custom_sort_key)

            write_sorted_header = False
            for d in sorted_data:

                if str(d[3]).lower().__contains__("Simulated Reality".lower()):
                    if include_srl is False:
                        continue

                # write header once
                if write_sorted_header is False:
                    write_header('sorted_generated_prompt.txt', d[1])
                    write_sorted_header = True

                # write body
                prompt_action_h = self.get_action_prompt(focus=d[4], game=sport)
                prompt_action_a = self.get_action_prompt(focus=d[5], game=sport)
                write_body('sorted_generated_prompt.txt', d[2].split("--")[0])
                write_action('sorted_generated_prompt.txt', prompt_action_h)
                write_body('sorted_generated_prompt.txt', d[2].split("--")[1])
                write_action('sorted_generated_prompt.txt', prompt_action_a)

                # write predict data prompt
                pdp = self.get_predict_prompt(d[4], d[5], d[3], sport=sport)
                # write_predict_data_prompt('predict_data_prompt.txt', pdp)
                write_predict_data_prompt('sorted_generated_prompt.txt', pdp)

            # open_file('generated_prompt.txt')
            self.open_file('sorted_generated_prompt.txt')
            # self.open_file('predict_data_prompt.txt')

            print("Completed!")
        else:
            self.start_single(self.single_home, self.single_away, self.single_league, sport=self.sport)
            pass
