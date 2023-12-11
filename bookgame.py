from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


def select_and_book_game(home_team, away_team):
    # Open SportyBet website
    driver = webdriver.Chrome()
    driver.get("https://sportybet.com/")

    # Search for the desired game
    search_bar = driver.find_element(By.ID, "search-bar")
    search_bar.send_keys(f"{home_team} vs {away_team}")
    search_bar.submit()

    # Wait for the game to be loaded
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, ".match-container"))
    )

    # Select the desired game
    game_container = driver.find_element(By.CSS_SELECTOR, ".match-container")
    game_container.click()

    # Wait for the betting options to be loaded
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, ".bet-option"))
    )

    # Select the desired betting option
    bet_option = driver.find_element(By.CSS_SELECTOR, ".bet-option")
    bet_option.click()

    # Enter the stake amount
    stake_input = driver.find_element(By.ID, "stake-input")
    stake_input.send_keys("100")

    # Place the bet
    place_bet_button = driver.find_element(By.ID, "place-bet-button")
    place_bet_button.click()

    # Confirm the bet
    confirm_bet_button = driver.find_element(By.ID, "confirm-bet-button")
    confirm_bet_button.click()

    # Close the browser window
    driver.quit()

home = "KF Besa Doberdoll"
away = "FK Teteks 1953"

select_and_book_game(home, away)