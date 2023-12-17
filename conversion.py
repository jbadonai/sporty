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
from selenium.webdriver.common.by import By
# from playwright.sync import sync_playwright


class Converter():
    def __init__(self):
        self.msport_url = "https://www.msport.com/ng/web/"

    def get_url_content(self, url):
        with sync_playwright() as p:
            browser = p.chromium.launch()
            context = browser.new_context()
            page = context.new_page()
            url = "your_website_url_here"
            page.goto(url)

            # Step 2: Find and click the link using Playwright
            link_selector = '.nav-item.router-link-active.active'
            page.click(link_selector)
        # print("loading Chrome...")
        # chrome_options = Options()
        # # chrome_options.add_argument("--headless")  # Run Chrome in headless mode (no GUI)
        # chrome_options.add_experimental_option('excludeSwitches', ['enable-logging'])
        #
        # # driver = webdriver.Chrome(options=chrome_options)  # Make sure chromedriver is in your PATH
        # driver = webdriver.Firefox()
        #
        # print(f"loading url [{url}]...")
        # driver.get(url)  # Load the page once
        # time.sleep(2)
        # content = driver.page_source
        #
        # return content, driver




    def start(self):
        content, driver = self.get_url_content(self.msport_url)
        time.sleep(2)
        css_selector = ".nav-item.router-link-active.active"
        driver.find_element(By.CSS_SELECTOR, css_selector).click()

        input('wait')

        bet_code_input = driver.find_element(By.CLASS_NAME, 'v-input')
        code = 'B2MZ68Y'
        bet_code_input.clear()
        bet_code_input.send_keys(code)



# c = Converter()
# c.start()

# Import selenium webdriver
from selenium import webdriver

# Create a driver object
driver = webdriver.Chrome()

# Assign the URL
url = "https://www.msport.com/ng/web/"

# Visit the page
driver.get(url)

# Find the home menu element by xpath
# home_menu = driver.find_element_by_xpath("//a[@class='nav-link' and text()='Home']")
home_menu = driver.find_element(By.XPATH, "//a[@class='nav-link' and text()='Home']")

# Click on the home menu
home_menu.click()

input('wait')