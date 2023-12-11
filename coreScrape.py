import requests
from bs4 import BeautifulSoup

def get_premier_league_data():
    # Fetch the Premier League data from the web
    url = "https://www.skysports.com/premier-league"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Extract the team data from the table
    teams_table = soup.find('table', id='table_standings')
    teams_rows = teams_table.find_all('tr')[1:]

    # Initialize an empty list to store team data
    team_data = []

    # Iterate through each team row and extract the relevant data
    for row in teams_rows:
        team_cells = row.find_all('td')[1:]
        team_name = team_cells[0].text.strip()
        team_is_home = False
        team_recent_5_performance = team_cells[3].text.strip()
        team_average_goals = float(team_cells[4].text.strip())
        team_player_injuries = int(team_cells[5].text.strip())
        team_possession_percentage = int(team_cells[6].text.strip())
        team_strength = int(team_cells[7].text.strip())
        head_to_head = team_cells[8].text.strip()
        opponent_team_name = team_cells[9].text.strip()
        opponent_is_home = True
        opponent_recent_5_performance = team_cells[10].text.strip()
        opponent_average_goals = float(team_cells[11].text.strip())
        opponent_player_injuries = int(team_cells[12].text.strip())
        opponent_possession_percentage = int(team_cells[13].text.strip())
        opponent_strength = int(team_cells[14].text.strip())
        opponent_head_to_head = team_cells[15].text.strip()
        match_outcome = team_cells[16].text.strip()

        # Append the extracted team data to the list
        team_data.append({
            "team_name": team_name,
            "team_is_home": team_is_home,
            "team_recent_5_performance": team_recent_5_performance,
            "team_average_goals": team_average_goals,
            "team_player_injuries": team_player_injuries,
            "team_possession_percentage": team_possession_percentage,
            "team_strength": team_strength,
            "head_to_head": head_to_head,
            "opponent_team_name": opponent_team_name,
            "opponent_is_home": opponent_is_home,
            "opponent_recent_5_performance": opponent_recent_5_performance,
            "opponent_average_goals": opponent_average_goals,
            "opponent_player_injuries": opponent_player_injuries,
            "opponent_possession_percentage": opponent_possession_percentage,
            "opponent_strength": opponent_strength,
            "opponent_head_to_head": opponent_head_to_head,
            "match_outcome": match_outcome
        })

    return team_data

# Print the extracted team data in a tabular format
team_data = get_premier_league_data()

print("| Team Name | Team is Home | Team Recent 5 Performance | Team Average Goals | Team Player Injuries | Team Possession Percentage | Team Strenght | Head to Head | Opponent Team Name | Opponent is Home | Opponent Recent 5 Performance | Opponent Average Goals | Opponent Player Injuries | Opponent Possession Percentage | Opponent Strenght | Head to Head | Match Outcome |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for team in team_data:
    print(f"| {team['team_name']} | {team['team_is_home']} | {team['team_recent_5_performance']} | {team['team_average_goals']} | {team['team_player_injuries']} | {team['team_possession_percentage']} | {team['team_strength']} | {team['head_to_head']} | {team['opponent_team_name']} | {team['opponent_is_home']} | {team['opponent_recent_5_performance']} | {team['opponent_average_goals']}")


