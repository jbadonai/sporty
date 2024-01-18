

class CalculateTeamStrength():
    def __init__(self):
        pass

    def calculate_team_strength(self, team_stats):
        # Define weights for each factor (you can adjust these based on importance)
        weights = {
            'win_loss_ratio': 0.25,
            'goals_for': 0.2,
            'goals_against': 0.2,
            'possession_percentage': 0.15,
            'pass_completion_rate': 0.1,
            'clean_sheets_ratio': 0.1
        }

        # Calculate weighted scores for each factor
        win_loss_score = team_stats.get('win_loss_ratio', 0) * weights['win_loss_ratio']
        goals_for_score = team_stats.get('goals_for', 0) * weights['goals_for']
        goals_against_score = team_stats.get('goals_against', 0) * weights['goals_against']
        possession_score = team_stats.get('possession_percentage', 0) * weights['possession_percentage']
        pass_completion_score = team_stats.get('pass_completion_rate', 0) * weights['pass_completion_rate']
        clean_sheets_score = team_stats.get('clean_sheets_ratio', 0) * weights['clean_sheets_ratio']

        # Calculate total strength percentage
        total_strength_percentage = (
                                            win_loss_score +
                                            goals_for_score +
                                            goals_against_score +
                                            possession_score +
                                            pass_completion_score +
                                            clean_sheets_score
                                    ) * 100

        return total_strength_percentage

    # # Sample data for two teams
    # team1_stats = {
    #     'win_loss_ratio': 0.7,
    #     'goals_for': 2.5,
    #     'goals_against': 1.0,
    #     'possession_percentage': 60,
    #     'pass_completion_rate': 85,
    #     'clean_sheets_ratio': 0.4
    # }
    #
    # team2_stats = {
    #     'win_loss_ratio': 0.6,
    #     'goals_for': 2.0,
    #     'goals_against': 1.2,
    #     'possession_percentage': 55,
    #     'pass_completion_rate': 80,
    #     'clean_sheets_ratio': 0.3
    # }
    #
    # # Calculate and print the strength percentage for each team
    # team1_strength = calculate_team_strength(team1_stats)
    # team2_strength = calculate_team_strength(team2_stats)
    #
    # print(f"Team 1 Strength: {team1_strength:.2f}%")
    # print(f"Team 2 Strength: {team2_strength:.2f}%")

c = CalculateTeamStrength()
