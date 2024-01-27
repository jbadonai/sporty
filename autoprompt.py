# try:
import subprocess
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
from datetime import datetime
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
import os
import requests
import pandas as pd
import time

class AutoPrompt():
    def __init__(self):
        pass

    def wait_for_element_by_xpath(self, driver, xpath):
        original_driver = driver
        counter = 0
        framecounter = 0
        framelist = driver.find_elements(By.TAG_NAME, "iframe")
        totalFrame = len(framelist)
        if len(framelist) > 1:
            print(f"Total Frame: {len(framelist)}")
            for f in framelist:
                print(f)

        while True:
            try:
                # driver.save_screenshot(f"{framecounter}.jpg")
                frames = driver.find_elements(By.TAG_NAME, "iframe")
                # print(f"internal: {len(frames)}")
                element = driver.find_element(By.XPATH, xpath)
                print(f"FOUND: {element.text}")
                return element

            except Exception as e:

                driver.switch_to.default_content()
                counter += 1
                if counter == 10:
                    if framecounter < totalFrame:
                        counter = 6
                        driver = original_driver
                        print(f'Working on frame {framecounter}')
                        driver.switch_to.frame(framelist[framecounter])
                        driver.save_screenshot(f"{framecounter}.jpg")
                        print(f"Switch to frame {framecounter} successful")
                        framecounter += 1
                    else:
                        print("No Further frame found. Element not found in all available frame...")

                print(f"\rRetrying [{counter}]...", flush=True, end='')
                time.sleep(1)
                continue

    def click_element(self, element):
        element.click()

    def load_url(self, url):
        print("loading Chrome...")
        chrome_options = Options()
        # chrome_options.add_argument("--headless")  # Run Chrome in headless mode (no GUI)
        chrome_options.add_experimental_option('excludeSwitches', ['enable-logging'])

        driver = webdriver.Chrome(options=chrome_options)  # Make sure chromedriver is in your PATH
        driver.get(url)  # Load the page once
        time.sleep(2)

        chat = self.wait_for_element_by_xpath(driver, '//*[@id="codex"]/a')
        self.click_element(chat)

        # time.sleep(5)
        later = self.wait_for_element_by_xpath(driver, '//*[@id="b_vfly_63997"]/div/div')
        self.click_element(later)

        driver.switch_to.default_content()

        # time.sleep(1)
        creative = self.wait_for_element_by_xpath(driver, '//*[@id="tone-options"]/li[1]/button')
        self.click_element(creative)

        input("::")


url = "https://www.bing.com/"
auto = AutoPrompt()
auto.load_url(url)

