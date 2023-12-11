team = [
    "USM Alger",
    "MC Alger",
    "MC Oran",
    "CS Constantine",
    "Ghali Club Mascara",
    "Rawd Solb Kouba",
    "USM El Harrach",
    "Nasr Air Algérie Hussein Dey",
    "USM Annaba",
    "MO Constantine",
    "US Chaouia",
    "ASO Chlef",
    "JS Saoura",
    "HB Chelghoum Laïd",
    "NC Magra",
    "Olympique de Médéa",
    "Paradou AC",
    "RC Arbaâ",
    "RC Relizane",
    "US Biskra",
    "WA Tlemcen",
    "ASO Chlef",
    "CR Belouizdad",
    "CS Constantine",
    "ES Sétif",
    "HB Chelghoum Laïd",
    "JS Kabylie",
    "JS Saoura",
    "MC Alger",
    "MC Oran",
    "NA Hussein Dey",
    "NC Magra",
    "Olympique de Médéa",
    "Paradou AC",
    "RC Arbaâ",
    "RC Relizane",
    "US Biskra",
    "USM Alger",
    "WA Tlemcen",
    "Aïn Témouchent Wilaya – Honor",
    "Aïn Témouchent Wilaya – Pre-Honor",
    "Mostaganem Wilaya – Honor",
    "Mostaganem Wilaya – Pre-Honor",
    "Oran Wilaya – Honor",
    "Oran Wilaya – Pre-Honor",
    "Relizane Wilaya – Honor",
    "Relizane Wilaya – Pre-Honor"
]

for t in team:
    text = f"in tabular form, list the last 10 games played by {t} with the following information:  " \
        f"team_home, team_away, goals_home,goals_away, possession_home, possession_away, shots_on_target_home," \
        f"shots_on_target_away\n\n"

    with open('prompts.txt', 'a') as f:
        f.write(text)

    print(text)









