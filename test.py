from selenium import webdriver
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager  # For managing driver
from bs4 import BeautifulSoup
import time

# Launch a Chrome browser
driver = webdriver.Chrome(ChromeDriverManager().install())

# Navigate to Bing Chat
driver.get("https://www.bing.com/chat/")

# Find the input field and enter your question
question_field = driver.find_element(By.ID, "question-input")
question_field.send_keys("Your question here")
question_field.submit()

# Wait for the response to appear (adjust wait time as needed)
time.sleep(5)  # Example wait time

# Get the HTML content of the response area
response_html = driver.find_element(By.ID, "response-area").get_attribute("innerHTML")

# Parse the HTML to extract the text response
soup = BeautifulSoup(response_html, "html.parser")
response_text = soup.get_text()

# Print the extracted response
print(response_text)

# Close the browser
driver.quit()