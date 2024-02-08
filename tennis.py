from PyQt5.QtWidgets import QMainWindow, QApplication, QSizeGrip,QLineEdit,QPushButton, \
    QMessageBox,QInputDialog, QVBoxLayout,  QSizePolicy, QGridLayout, QLabel, QProgressBar, QProgressDialog
from PyQt5.QtCore import Qt, QPropertyAnimation, QEasingCurve, QBasicTimer
from PyQt5.QtGui import QIcon
from PyQt5 import QtCore
from ui import tennis_
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import requests
from bs4 import BeautifulSoup
from requests_html import HTMLSession

class TennisWeb():
    def __init__(self):
        self.url = r"https://paripesa.ng/en/live/tennis"

        chrome_options = Options()
        # chrome_options.add_argument('--headless')  # Run Chrome in headless mode
        self.driver = webdriver.Chrome(options=chrome_options)
        self.main_window_handle = None

        pass

    def goto(self):
        self.driver.get("https://paripesa.ng/en/live/tennis")
        time.sleep(5)

    def click_statistics_button(self, driver):
        # Locate the statistics button by class name
        # Wait for the statistics button to be present
        print('waiting....')
        wait = WebDriverWait(driver, 500)
        statistics_button = wait.until(EC.presence_of_element_located((By.CLASS_NAME, 'ui-dashboard-game-button')))

        # Click on the statistics button
        # statistics_button.click()
        print('done waiting...')
        statistics_button = driver.find_elements(By.CLASS_NAME, 'ui-dashboard-game-button')
        time.sleep(2)
        statistics_button[1].click()



    def switch_to_popup_window(self, driver):
        try:
            # Switch to the new window
            main_window = driver.current_window_handle
            self.main_window_handle = main_window
            new_window_handle = None

            # Wait for the new window to open
            wait = WebDriverWait(driver, 10)
            wait.until(lambda driver: len(driver.window_handles) > 1)
            # print(f'total winows: {len(driver.window_handles)}')

            # Switch to the new window
            for window_handle in driver.window_handles:
                # print(f"window handle :{window_handle}")
                if window_handle != main_window:
                    new_window_handle = window_handle
                    # print(f"new window handle: {new_window_handle}")
                    break

            driver.switch_to.window(new_window_handle)
            # time.sleep(20)

            current_url = driver.current_url
            time.sleep(2)
            driver.switch_to.window(main_window)
            time.sleep(2)
            driver.switch_to.window(new_window_handle)
            time.sleep(2)
            driver.get(current_url)
            print('im here')

            time.sleep(120)

            # players = self.get_players_name(current_url)
            # print(players)
        except Exception as e:
            print(f"An error occured in switch to popup window: {e}")
            pass

        # print('waiting for live....')
        # while True:
        #     try:
        #         liveWait = WebDriverWait(driver, 10)
        #         # liveStat = liveWait.until(EC.presence_of_element_located((By.CLASS_NAME, 'old-switch__text')))
        #         liveStat = liveWait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="__layout"]/div/div[1]/div[1]/div[2]/div[2]/a[1]/div')))
        #         break
        #     except Exception as e:
        #         print('retrying')
        #         driver.refresh()
        #         time.sleep(10)
        #         continue

    def get_players_name(self, url):
        self.driver.get(url)






    def check_match_history_in_popup(self, driver):
        # Locate all elements with class name 'old-section__title' within the pop-up
        titles = driver.find_elements(By.CLASS_NAME, 'old-section__title')
        print(f'total titles = {len(titles)}')

        # Check if any title contains the text 'Match History'
        for title in titles:
            print(title)
            if 'MATCH STATISTICS' in title.text:
                return True

        # If 'Match History' is not found in any title, return False
        return False


class Tennis(QMainWindow, tennis_.Ui_MainWindow):

    def __init__(self):
        super(Tennis, self).__init__()
        self.setupUi(self)
        self.buttonPredict.clicked.connect(self.get_data_from_web)
        self.setWindowFlags(Qt.WindowStaysOnTopHint)
        self.label_result2.setVisible(False)
        self.textAWin2.textChanged.connect(self.auto_start)
        self.textAWin1.textChanged.connect(self.auto_start)
        self.textABreakPoint.textChanged.connect(self.auto_start)
        self.textBWin1.textChanged.connect(self.auto_start)
        self.textBWin2.textChanged.connect(self.auto_start)
        self.textBBreakPoint.textChanged.connect(self.auto_start)

        self.tennisWeb = TennisWeb()
        # self.buttonPredict.setVisible(False)


    def calculate_service_win_probability(self, win_1st_serve, win_2nd_serve, break_point_save):
        p_win_1st_serve = win_1st_serve / 100
        p_win_2nd_serve = win_2nd_serve / 100
        p_break_point_save = break_point_save / 100

        p_win = p_win_1st_serve + (1 - p_win_1st_serve) * p_win_2nd_serve * p_break_point_save

        return p_win

    # Define a function to calculate the probability of winning a point on serve
    def p(self, w1, w2):
        return (w1 + (1 - w1) * w2) / (1 + (1 - w1) * w2)

    # Define a function to calculate the probability of winning the service
    def P(self, p):
        return p ** 4 + 4 * p ** 4 * (1 - p) + 10 * p ** 4 * (1 - p) ** 2 + 20 * p ** 3 * (1 - p) ** 3

    def option2(self,w1_A, w2_A, w1_B, w2_B):

        # Calculate the probability of winning a point on serve for each team
        p_A = self.p(w1_A, w2_A)
        p_B = self.p(w1_B, w2_B)

        # Calculate the probability of winning the service for each team
        P_A = self.P(p_A)
        P_B = self.P(p_B)

        # Print the results
        result = f"""
        The probability of team A winning its service is {P_A:.2f}
        The probability of team B winning its service is {P_B:.2f}
        """
        self.label_result2.setText(result)

    def auto_start(self):
        sender = self.sender()

        if sender.objectName() == self.textABreakPoint.objectName() or \
            sender.objectName() == self.textAWin1.objectName() or  \
            sender.objectName() == self.textAWin2.objectName() or \
            sender.objectName() == self.textBBreakPoint.objectName() or \
            sender.objectName() == self.textBWin1.objectName() or \
                sender.objectName() == self.textBWin2.objectName():

            self.start()

    def get_data_from_web(self):
        # goto tennis page
        print('going to tennis webpage...')
        self.tennisWeb.goto()
        # click on statistic button
        print('clicking the statistics button...')
        self.tennisWeb.click_statistics_button(self.tennisWeb.driver)
        # waiting for page to load
        print('waiting for statistics page to load')
        time.sleep(2)
        # switch to pup up window
        print('switching to statistic pop up window')
        self.tennisWeb.driver = self.tennisWeb.switch_to_popup_window(self.tennisWeb.driver)
        # search the popup window
        print('checking for history....')
        result = self.tennisWeb.check_match_history_in_popup(self.tennisWeb.driver)
        print(f"Result: {result}")

    def start(self):

        try:
            if self.textAWin1.text() == "":
                self.textAWin1.setText("0")
            if self.textAWin2.text() == "":
                self.textAWin2.setText("0")
            if self.textABreakPoint.text() == "":
                self.textABreakPoint.setText("0")
            if self.textBWin1.text() == "":
                self.textBWin1.setText("0")
            if self.textBWin2.text() == "":
                self.textBWin2.setText("0")
            if self.textBBreakPoint.text() == "":
                self.textBBreakPoint.setText("0")

            a1 = int(self.textAWin1.text())
            a2 = int(self.textAWin2.text())
            ab = int(self.textABreakPoint.text())

            b1 = int(self.textBWin1.text())
            b2 = int(self.textBWin2.text())
            bb = int(self.textBBreakPoint.text())

            if a1 == "" or a2 == "" or ab == "" or b1 == "" or b2 == "" or bb == "":

                raise Exception

            team_a_statistics = {'win_1st_serve': a1, 'win_2nd_serve': a2, 'break_point_save': ab}
            team_b_statistics = {'win_1st_serve': b1, 'win_2nd_serve': b2, 'break_point_save': bb}

            # Calculate probabilities for each team
            p_win_team_a = self.calculate_service_win_probability(**team_a_statistics)
            p_win_team_b = self.calculate_service_win_probability(**team_b_statistics)

            # Print the results
            result = f"""
            Probability of Team A winning the service: {p_win_team_a:.2%}"
            Probability of Team B winning the service: {p_win_team_b:.2%}
            """
            self.label_result.setText(result)

            # self.option2(a1, a2, b1, b2)
        except Exception as e:
            print(e)


if __name__ == '__main__':
    app = QApplication([])
    app.setStyle('fusion')

    win = Tennis()
    win.show()
    app.exec_()
