from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from bs4 import BeautifulSoup
import csv
import time

#https://chromedriver.storage.googleapis.com/index.html?path=114.0.5735.90/
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument("--lang=en-GB")
driver = webdriver.Chrome(options=chrome_options) 

# Open the URL
url = 'https://www.google.com/maps/place/%D9%85%D8%B7%D8%B9%D9%85+%D9%85%D8%B2%D9%8A%D8%A7%D9%86+-+Mezyan%E2%80%AD/@33.8955174,35.4840395,15z/data=!4m18!1m9!3m8!1s0x151f17298f36406d:0x81e65f19355c7ff3!2z2YXYt9i52YUg2YXYstmK2KfZhiAtIE1lenlhbg!8m2!3d33.8955174!4d35.4840395!9m1!1b1!16s%2Fg%2F11c40ptll4!3m7!1s0x151f17298f36406d:0x81e65f19355c7ff3!8m2!3d33.8955174!4d35.4840395!9m1!1b1!16s%2Fg%2F11c40ptll4?entry=ttu'
driver.get(url)

# Wait for the page elements to load
wait = WebDriverWait(driver, 10)
menu_bt = wait.until(EC.element_to_be_clickable(
                       (By.XPATH, '//button[@data-value=\'Sort\']'))
                   )  
menu_bt.click()

time.sleep(5)
#wait = WebDriverWait(driver, 5)

# Perform the sequence of key presses
action = ActionChains(driver)
action.send_keys(Keys.ARROW_DOWN).pause(1)   # Press Arrow Down
action.send_keys(Keys.ARROW_UP).pause(1)     # Press Arrow Up
action.send_keys(Keys.ENTER).perform()       # Press Enter
time.sleep(5)

# Calculate the number of presses needed
duration = 600  # duration in seconds to press the End key ADJUSTED BASED ON INTERNET SPEED
interval = 1    # interval in seconds between each press
num_presses = duration // interval

# Perform the sequence of pressing End key
for _ in range(num_presses):
    action = ActionChains(driver)
    action.send_keys(Keys.END).perform()
    time.sleep(interval)

# Wait an additional 5 seconds after the loop
time.sleep(5)

# Find the 'More' buttons and click them
more_buttons = driver.find_elements(By.XPATH, '//button[text()="More"]')
for button in more_buttons:
    try:
        button.click()
        time.sleep(2)  # Small delay to allow content to load
    except Exception as e:
        print("Error clicking button:", e)

# Process and store reviews
def get_review_summary(result_set):
    reviews = []
    for result in result_set:
        review_name = result.find(class_='d4r55').text
        review_rating = result.find('span', {'class':'kvMYJc'})['aria-label']
        
        # Check if the review text element exists
        review_text_element = result.find('span', class_='wiI7pd')
        if review_text_element:
            review_text = review_text_element.text.strip()
        else:
            review_text = ""

        reviews.append({'Review Name': review_name, 'Review Rating': review_rating, 'Review Text': review_text})
    return reviews


response = BeautifulSoup(driver.page_source, 'html.parser')
review_elements = response.find_all('div', class_='jftiEf')
reviews = get_review_summary(review_elements)

# Save reviews to a CSV file
reviews_scraped = 'reviews.csv'
with open(reviews_scraped, 'w', newline='', encoding='utf-8') as file:
    writer = csv.DictWriter(file, fieldnames=['Review Name', 'Review Rating', 'Review Text'])
    writer.writeheader()
    for review in reviews:
        writer.writerow(review)

# Quit
input("Press Enter to close the browser")
driver.quit()
