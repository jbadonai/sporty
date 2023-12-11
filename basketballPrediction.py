# def calculate_points_per_possession(team_stats, weights):
#     efg_percentage = team_stats['efg_percentage']
#     turnover_rate = team_stats['turnover_rate']
#     offensive_rebound_rate = team_stats['offensive_rebound_rate']
#     free_throw_rate = team_stats['free_throw_rate']
#
#     points_per_possession = (
#         weights['efg'] * efg_percentage +
#         weights['tov'] * (1 - turnover_rate) +
#         weights['orb'] * offensive_rebound_rate +
#         weights['ftr'] * free_throw_rate
#     )
#
#     return points_per_possession
#
#
# def predict_scores(team1_stats, team2_stats):
#     weights = {
#         'efg': 0.4,
#         'tov': 0.25,
#         'orb': 0.2,
#         'ftr': 0.15
#     }
#
#     team1_points_per_possession = calculate_points_per_possession(team1_stats, weights)
#     team2_points_per_possession = calculate_points_per_possession(team2_stats, weights)
#
#     # Assuming both teams have the same number of possessions
#     possessions = 100
#
#     team1_predicted_score = team1_points_per_possession * possessions
#     team2_predicted_score = team2_points_per_possession * possessions
#
#     return team1_predicted_score, team2_predicted_score
#
#
# # Example usage:
# team1_stats = {
#     'efg_percentage': 0.55,
#     'turnover_rate': 0.12,
#     'offensive_rebound_rate': 0.25,
#     'free_throw_rate': 0.2
# }
#
# team2_stats = {
#     'efg_percentage': 0.52,
#     'turnover_rate': 0.15,
#     'offensive_rebound_rate': 0.22,
#     'free_throw_rate': 0.18
# }
#
# predicted_scores = predict_scores(team1_stats, team2_stats)
# print(f"Team 1 Predicted Score: {predicted_scores[0]:.2f}")
# print(f"Team 2 Predicted Score: {predicted_scores[1]:.2f}")

# Import pandas and numpy libraries
import pandas as pd
import numpy as np

'''
Effective Field Goal Percentage (efg) = (FG + 0.5 * 3P) / FGA
Turnover Percentage (tov) = TOV / (FGA + 0.44 * FTA + TOV)
Offensive Rebound Percentage (orb) = ORB / (ORB + Opp DRB)
Free Throw Rate (ftr) = FT / FGA
To calculate the number of possessions for each team, you can use this formula:

Possessions (poss) = 0.5 * [ (Team FGA) + 0.4 * (Team FTA) - 1.07 * (Team ORB / (Team ORB + Opp DRB)) * (Team FGA - Team FG) + Team TOV] + 0.5 * [ (Opp FGA) + 0.4 * (Opp FTA) - 1.07 * (Opp ORB / (Opp ORB + Team DRB)) * (Opp FGA - Opp FG) + Opp TOV]
'''


class Predictor():

    def __init__(self):
        self.home_fga = None    # Field goal attempt
        self.home_fg = None     # field goal made
        self.home_3p = None     # 3 pointer made
        self.home_orb = None    # offensive rebound
        self.home_drb = None    # defensive rebound
        self.home_tov = None    # turnover
        self.home_fta = None    # free throw attempt
        self.home_ft = None     # free throw made
        self.away_fga = None
        self.away_fg = None
        self.away_3p = None
        self.away_orb = None
        self.away_drb = None
        self.away_tov = None
        self.away_fta = None
        self.away_ft = None

        self.home_efg = None    # effective goal percentage
        self.home_tovp = None    # turnover percentage
        self.home_orbp = None   # offensive rebound percentage
        self.home_ftr = None    # free throw rate
        self.home_poss = None   # possession
        self.away_efg = None    # effective goal percentage
        self.away_tovp = None    # turnover percentage
        self.away_orbp = None   # offensive rebound percentage
        self.away_ftr = None    # free throw rate
        self.away_poss = None   # possession

    def get_data(self, data=None):
        data = {
          "home": {
                "FGA": 57.8,
                "FG": 25.4,
                "3P": 7.6,
                "ORB": 9.8,
                "DRB": 23.2,
                "TOV": 13.8,
                "FTA": 19.4,
                "FT": 14.8
            },
          "away": {
                "FGA": 60.6,
                "FG": 28.6,
                "3P": 10.8,
                "ORB": 8.6,
                "DRB": 24.4,
                "TOV": 11.6,
                "FTA": 18.8,
                "FT": 14.6
            }
        }

        self.home_fga = data['home']['FGA']
        self.home_fg = data['home']['FG']
        self.home_3p = data['home']['3P']
        self.home_orb = data['home']['ORB']
        self.home_drb = data['home']['DRB']
        self.home_tov = data['home']['TOV']
        self.home_fta = data['home']['FTA']
        self.home_ft = data['home']['FT']
        self.away_fga = data['away']['FGA']
        self.away_fg = data['away']['FG']
        self.away_3p = data['away']['3P']
        self.away_orb = data['away']['ORB']
        self.away_drb = data['away']['DRB']
        self.away_tov = data['away']['TOV']
        self.away_fta = data['away']['FTA']
        self.away_ft = data['away']['FT']

    def calculate_parameters(self):
        try:
            # EFG
            self.home_efg = self.calculate_efg(self.home_fg, self.home_3p, self.home_fga)
            self.away_efg = self.calculate_efg(self.away_fg, self.away_3p, self.away_fga)

            # TOV percentage
            self.home_tovp = self.calculate_tovp(self.home_tov,self.home_fga, self.home_fta)
            self.away_tovp = self.calculate_tovp(self.away_tov,self.away_fga, self.away_fta)

            # ORB percentage
            self.home_orbp = self.calculate_orbp(self.home_orb, self.away_drb)
            self.away_orbp = self.calculate_orbp(self.away_orb, self.home_drb)

            # FTR
            self.home_ftr = self.calculate_ftr(self.home_ft, self.home_fga)
            self.away_ftr = self.calculate_ftr(self.away_ft, self.away_fga)

            # possession
            self.calculate_possessions()
            pass
        except Exception as e:
            print(f"An Error occurred in calculate parameters: {e}")
            pass

    def calculate_efg(self, fg, threep, fga):
        efg = (fg + 0.5 * threep)/fga
        print(f"efg: {efg}")
        return efg

    def calculate_tovp(self, TOV, FGA, FTA):
        tovp = TOV / (FGA + 0.44 * FTA + TOV)
        print(f"tovp: {tovp}")
        return tovp

    def calculate_orbp(self, team_orb, opp_drb):
        orbp = team_orb / (team_orb + opp_drb)
        print(f"orbp: {orbp}")
        return orbp

    def calculate_ftr(self,FT, FGA):
        ftr = FT / FGA
        print(f"ftr: {ftr}")
        return  ftr

    def calculate_possessions(self):
        self.home_poss = 0.5 * (self.home_fga + 0.4 * self.home_fta - 1.07 * (self.home_orb / (self.home_orb + self.away_drb))
                           * (self.home_fga - self.home_fg) + self.home_tov)

        self.away_poss = 0.5 * (self.away_fga + 0.4 * self.away_fta - 1.07 * (self.away_orb / (self.away_orb + self.home_drb))
                           * (self.away_fga - self.away_fg) + self.away_tov)

        total_possessions = self.home_poss + self.away_poss
        print(f"poss: {self.home_poss}: {self.away_poss}")
        return total_possessions

    # Define a function to calculate the points per possession for a team
    def ppp(self, efg, tov, orb, ftr):
      # Use the weights suggested by Oliver
      efg_weight = 0.4
      tov_weight = 0.25
      orb_weight = 0.2
      ftr_weight = 0.15

      # Calculate the points per possession
      return efg_weight * efg + tov_weight * (1 - tov) + orb_weight * orb + ftr_weight * ftr


    # Define a function to predict the scores between two teams
    def predict_scores(self):
      # Calculate the points per possession for each team
      team1_ppp = self.ppp(self.home_efg, self.home_tov, self.home_orb, self.home_ftr)
      team2_ppp = self.ppp(self.away_efg, self.away_tov, self.away_orb, self.away_ftr)
      print(f"team 2 adn 2 poss: {team1_ppp} --- {team2_ppp}")

      # Calculate the expected scores for each team
      team1_score = self.home_poss * team1_ppp
      team2_score = self.away_poss * team2_ppp

      # Return the predicted scores as a tuple
      return (team1_score, team2_score)


# # Example: Predict the scores between Lakers and Celtics
# lakers_score, celtics_score = predict_scores("Lakers", "Celtics")
# print(f"The predicted score for Lakers is {lakers_score}")
# print(f"The predicted score for Celtics is {celtics_score}")

p = Predictor()
p.get_data()
p.calculate_parameters()
home, away = p.predict_scores()

print(f"home: {home}")
print(f"away: {away}")

