##############################################################################################################

''' This code is being used to perform web scraping on Airbnb data from downtown Toronto, 
specifically for any given week and a booking of 2 guests. The scraped data is then stored in an Excel (.csv) 
file for further analysis or reference. '''

##############################################################################################################

# Import required libraries
import re
import time
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
import nltk
from nltk import pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
import seaborn as sns

''' Scrape data using BeautifulSoup and Selenium packages '''

# Retrieve data from Airbnb's website in Downtown Toronto for a stay for 2 guests (any week)
initial_url = 'https://www.airbnb.ca/s/Downtown-Toronto--Toronto--Ontario--Canada/homes?tab_id=home_tab&refinement_paths%5B%5D=%2Fhomes&flexible_trip_lengths%5B%5D=one_week&monthly_start_date=2023-11-01&monthly_length=3&price_filter_input_type=0&price_filter_num_nights=5&channel=EXPLORE&query=Downtown%20Toronto%2C%20Toronto%2C%20ON&place_id=ChIJvRBz0jTL1IkRkwMHIgbSFbo&date_picker_type=calendar&adults=2&source=structured_search_input_header&search_type=autocomplete_click'

# Send an HTTP GET request to the specified URL
response = requests.get(initial_url)

# Print the response object to check the status and content (200 means request was successfull)
print(response)

# Use BeautifulSoup to parse the HTML content of the initial URL
soup = BeautifulSoup(requests.get(initial_url).content, 'html.parser')

# Find all meta tags with itemprop="url" and extract URLs
url_meta_tags = soup.find_all("meta", itemprop="url")
fixed_urls = ["https://" + tag.get("content") for tag in url_meta_tags]

# Initialize first url
url = initial_url

# Retrieve all Airbnb posts (18) from each page (15)
while True:
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'lxml')
    next_page_element = soup.find('a', class_='l1ovpqvx c1ytbx3a dir dir-ltr')

    # Check if there is no next page link (end of search results).
    if next_page_element is None:
        break  # Exit the loop if there is no next page.

    # Extract the 'href' attribute from the next page element and form the complete URL.
    next_page = next_page_element.get('href')
    complete_next_page = "https://www.airbnb.ca/" + next_page

    soup = BeautifulSoup(requests.get(complete_next_page).content, 'html.parser')

    # Find all meta tags with itemprop="url" and extract URLs
    url_meta_tags = soup.find_all("meta", itemprop="url")
    fixed_urls_temp = ["https://" + tag.get("content") for tag in url_meta_tags]

    fixed_urls.extend(fixed_urls_temp)

    # Update the 'url' variable to continue the loop with the next page.
    url = complete_next_page

# Verify that all urls are stored (18*15=270)
len(fixed_urls)

''' Required functions for web scraping. '''

def text_to_list(text):
    # Split the text into lines and create a list
    items = text.split("\n")

    # Remove any empty lines if present
    items = [item.strip() for item in items if item.strip()]

    return items

def extract_number_from_text(text):
    # Use regular expressions to extract the number
    matches = re.findall(r'\d+', text)

    if matches:
        number = int(matches[0])
        return number
    else:
        return None

def extract_rating_reviews_from_text(text):
    # Use a regular expression to extract numbers
    numbers = re.findall(r'\d+\.\d+|\d+', text)

    # Convert the extracted strings to float or int as needed
    numbers = [float(number) if '.' in number else int(number) for number in numbers]

    return numbers

def extract_first_number(input_string):
    # Use a regular expression to search for the first decimal number in the input string
    match = re.search(r'(\d+\.\d+)', input_string)
    if match:
        number = match.group(1)
        return number
    else:
        return None
    
def extract_number(input_string):
    # Use a regular expression to search for a number (integer or decimal) in the input string
    match = re.search(r'(\d+(\.\d+)?)', input_string)
    if match:
        number = float(match.group(1))
        return number
    else:
        return None
    
# Initialize lists
title_col = []
summary_col = []
price_col = []
sale_price_col = []
amenities_col = []
num_amenities_col = []
basic_info_col = []
rating_reviews_col = []
rating_breakdown_col = []

# Loops through each url and store information in appropriate list
for url in fixed_urls:
    driver = webdriver.Chrome()
    driver.get(url)
    
    # Wait for the page to load (you may need to adjust the waiting time)
    time.sleep(20)

    try:
        # Extract information using Selenium
        title = driver.find_element(By.XPATH,'//*[@id="site-content"]/div/div[1]/div[1]/div[1]/div/div/div/div/div/section/div[1]/span/h1').text
        title_col.append(title)
    except NoSuchElementException:
        title_col.append('N/A')

    try:
        # Extract information using Selenium
        summary = driver.find_element(By.XPATH,'//*[@id="site-content"]/div/div[1]/div[3]/div/div[1]/div/div[1]/div/div/section/div/div/div[1]/div/h2').text
        summary_col.append(summary)
    except NoSuchElementException:
        summary_col.append('N/A')

    try:
        # Extract information using Selenium
        price = driver.find_element(By.XPATH,'//*[@id="site-content"]/div/div[1]/div[3]/div/div[2]/div/div/div[1]/div/div/div/div/div/div/div/div[1]/div[1]/div[1]/div/span/div/span[1]').text
        price_col.append(price)
    except NoSuchElementException:
        price_col.append('N/A')

    try:
        # Extract information using Selenium
        sale_price = driver.find_element(By.XPATH,'//*[@id="site-content"]/div/div[1]/div[3]/div/div[2]/div/div/div[1]/div/div/div/div/div/div/div/div[1]/div[1]/div[1]/div/span/div/span[2]').text

        # Check if sale_price is 'night' and store 'N/A' if it is
        if sale_price.strip() == 'night':
            sale_price_col.append('N/A')
        else:
            sale_price_col.append(sale_price)
    except NoSuchElementException:
        sale_price_col.append('N/A')

    try:
        # Extract information using Selenium
        basic_info = driver.find_element(By.XPATH,'//*[@id="site-content"]/div/div[1]/div[3]/div/div[1]/div/div[1]/div/div/section/div/div/div[1]/ol').text
        basic_info_col.append([item.strip() for item in basic_info.split("·")])
    except NoSuchElementException:
        basic_info_col.append('N/A')

    try:
        amenities = driver.find_element(By.XPATH,'//*[@id="site-content"]/div/div[1]/div[3]/div/div[1]/div/div[5]/div/div[2]/section/div[3]').text
        amenities_col.append(text_to_list(amenities))
    except NoSuchElementException:
        amenities_col.append('N/A')

    try:
        num_amenities = driver.find_element(By.XPATH,'//*[@id="site-content"]/div/div[1]/div[3]/div/div[1]/div/div[5]/div/div[2]/section/div[4]').text
        num_amenities_col.append(extract_number_from_text(num_amenities))
    except NoSuchElementException:
        num_amenities_col.append('N/A')

    try:
        overall_rating = driver.find_element(By.XPATH,'//*[@id="site-content"]/div/div[1]/div[3]/div/div[2]/div/div/div[1]/div/div/div/div/div/div/div/div[1]/div[1]/div[2]/span/span[2]').text
        overall_rating = extract_number(overall_rating)

        num_reviews = driver.find_element(By.XPATH,'//*[@id="site-content"]/div/div[1]/div[3]/div/div[2]/div/div/div[1]/div/div/div/div/div/div/div/div[1]/div[1]/div[2]/span/span[3]').text
        num_reviews = int(extract_number(num_reviews))

        rating_reviews_col.append([overall_rating,num_reviews])
    except NoSuchElementException:
        rating_reviews_col.append('N/A')

    try:
        cleanliness = driver.find_element(By.XPATH,'//*[@id="site-content"]/div/div[1]/div[4]/div/div/div/div[2]/div/section/div[2]/div/div/div[3]/div/div/div/div/div[2]/div/div').text
        cleanliness = extract_first_number(cleanliness)

        accuracy = driver.find_element(By.XPATH,'//*[@id="site-content"]/div/div[1]/div[4]/div/div/div/div[2]/div/section/div[2]/div/div/div[3]/div/div/div/div/div[3]/div/div').text
        accuracy = extract_first_number(accuracy)

        checkin = driver.find_element(By.XPATH,'//*[@id="site-content"]/div/div[1]/div[4]/div/div/div/div[2]/div/section/div[2]/div/div/div[3]/div/div/div/div/div[4]/div/div').text
        checkin = extract_first_number(checkin)

        communication = driver.find_element(By.XPATH,'//*[@id="site-content"]/div/div[1]/div[4]/div/div/div/div[2]/div/section/div[2]/div/div/div[3]/div/div/div/div/div[5]/div/div').text
        communication = extract_first_number(communication)

        location = driver.find_element(By.XPATH,'//*[@id="site-content"]/div/div[1]/div[4]/div/div/div/div[2]/div/section/div[2]/div/div/div[3]/div/div/div/div/div[6]/div/div').text
        location = extract_first_number(location)

        value = driver.find_element(By.XPATH,'//*[@id="site-content"]/div/div[1]/div[4]/div/div/div/div[2]/div/section/div[2]/div/div/div[3]/div/div/div/div/div[7]/div/div').text
        value = extract_first_number(value)

        rating_breakdown_col.append([cleanliness,accuracy,checkin,communication,location,value])
    except NoSuchElementException:
        rating_breakdown_col.append('N/A')

    # Close the Selenium webdriver
    driver.quit()

# Create a DataFrame
df = pd.DataFrame({'Title': title_col,'Summary': summary_col, 'Price per night': price_col, 'Sale price': sale_price_col,'Basic Info':basic_info_col, 'Amenities': amenities_col, 'Number of Amenities': num_amenities_col,'Rating and Number of Reviews':rating_reviews_col,'Rating Breakdown':rating_breakdown_col})

# Display the DataFrame
print(df)

# Display dimensions of DataFrame
df.shape

''' Drop any duplicate rows. We use the columns 'Title' and 'Summary' because these two columns contain
unique words set by the owner of the dwelling, and they do not contain lists which are not compatible
with the drop function. '''

# Check for duplicate rows
df.duplicated(subset=['Title','Summary']).sum()

# Drop one of the duplicate rows
df = df.drop_duplicates(subset=['Title','Summary'])

# Display dimensions of DataFrame
df.shape

''' Clean 'Price per night' and 'Sale price' columns. '''

# Remove "$" and " CAD" from 'Price per night' column, ignoring N/A
cleaned_data = []

for value in df['Price per night']:
    if value != "N/A":
        cleaned_value = int(value.replace("$", "").replace(" CAD", ""))
        cleaned_data.append(cleaned_value)
    else:
        cleaned_data.append('N/A')

# Update 'Price per night' column
df['Price per night'] = cleaned_data
df = df.rename(columns={'Price per night': 'Price per night (CAD)'}) # Rename the column

# Remove "$" and " CAD" from 'Sale price' column, ignoring N/A
cleaned_data = []

for value in df['Sale price']:
    if value != "N/A":
        cleaned_value = int(value.replace("$", "").replace(" CAD", ""))
        cleaned_data.append(cleaned_value)
    else:
        cleaned_data.append('N/A')

# Update 'Sale price' column
df['Sale price'] = cleaned_data
df = df.rename(columns={'Sale price': 'Sale price (CAD)'}) # Rename the column

''' Clean 'Basic Info' column. Split each list into a column for each item, that is, 'Number of guests', 
'Number of bedrooms', 'Number of beds', and 'Number of bathrooms'. Replace missing values with 'N/A' for
consistency.  '''

# Split the 'Basic Info' into multiple columns
split_columns = df['Basic Info'].apply(pd.Series)

# Rename the new columns
split_columns = split_columns.rename(columns={0: 'Number of guests', 1: 'Number of bedrooms', 2: 'Number of beds',3:'Number of bathrooms'})

# Concatenate the original DataFrame with the new columns
result_df = pd.concat([df, split_columns], axis=1)

# Drop column 'Basic Info'
result_df = result_df.drop('Basic Info', axis=1)

''' Remove text from 'Number of guests', 'Number of bedrooms', 'Number of beds', and 'Number of bathrooms' 
columns and store only numbers.'''

# Extract numbers from the 'Text' column
result_df['Number of guests'] = result_df['Number of guests'].str.extract('(\d+)')
result_df['Number of beds'] = result_df['Number of beds'].str.extract('(\d+)')
result_df['Number of bathrooms'] = result_df['Number of bathrooms'].apply(lambda x: None if x == 'N/A' else re.findall(r'\d+\.\d+|\d+', str(x)))

# Extract numbers while handling 'N/A' and 'Studio'
def extract_numbers(text):
    if text == 'N/A':
        return None
    elif text == 'Studio':
        return 0
    else:
        try:
            # Extract numbers using regular expression
            number = int(re.search('(\d+)', text).group())
            return number
        except AttributeError:
            # No numbers found
            return None

# Apply the function to the 'Text' column
result_df['Number of bedrooms']=result_df['Number of bedrooms'].apply(extract_numbers)

# Replace NaN with 'N/A' for consistency
result_df['Number of guests'] = result_df['Number of guests'].fillna('N/A')
result_df['Number of bedrooms'] = result_df['Number of bedrooms'].fillna('N/A')
result_df['Number of beds'] = result_df['Number of beds'].fillna('N/A')
result_df['Number of bathrooms'] = result_df['Number of bathrooms'].fillna('N/A')

# Custom function to convert lists to individual values
def convert_lists(value):
    if isinstance(value, list):
        return value[0]
    else:
        return value

# Apply the function to the 'Number of bathrooms' column
result_df['Number of bathrooms'] = result_df['Number of bathrooms'].apply(convert_lists)

''' Seperate 'Unavailable' amenities from 'Amenities' column of its own column. Remove unavailable amenities 
from 'Amenities' column. Rename columns 'Amenities shown' and 'Unavailable amenities shown', by which we 
mean shown on the main post without clicking 'View all amenities'. '''

# Rename the column
result_df = result_df.rename(columns={'Amenities': 'Amenities shown'})
result_df = result_df.rename(columns={'Number of Amenities': 'Number of amenities'})

# Make column with unavailable item shown on main post
def process_list(lst):
    for item in lst:
        if item.startswith('Unavailable:'):
            return item[len('Unavailable:'):]
    return 'N/A'

# Apply the custom function to create the new column
result_df['Unavailable amenities shown'] = result_df['Amenities shown'].apply(process_list)

# Create new column of unavailable amenities shown
def process_list(lst):
    unavailable_items = []
    for item in lst:
        if item.startswith('Unavailable:'):
            unavailable_items.append(item[len('Unavailable:'):])
    if unavailable_items:
        return unavailable_items
    return ['N/A']

# Apply the function to create the new column
result_df['Unavailable amenities shown'] = result_df['Amenities shown'].apply(process_list)
result_df['Unavailable amenities shown'] = result_df['Unavailable amenities shown'].apply(lambda x: 'N/A' if x == ['N/A'] else x)

# Function to remove items containing 'Unavailable:...'
def remove_unavailable_items(lst):
    return [item for item in lst if not item.startswith('Unavailable:')]

# Apply the custom function to create the new column
result_df['Amenities shown'] = result_df['Amenities shown'].apply(remove_unavailable_items)

''' Clean up '[N, /, A]'  in 'Amenities shown' column caused by missing values. '''

# Function to replace lists of the form [N, /, A] with 'N/A'
def replace_in_list(lst):
    return 'N/A' if lst == ['N', '/', 'A'] else lst

# Apply the custom function to the column
result_df['Amenities shown'] = result_df['Amenities shown'].apply(replace_in_list)

''' Change order of columns. '''

# Define the desired column order
new_order = ['Title','Summary','Price per night (CAD)','Sale price (CAD)','Number of guests','Number of bedrooms','Number of beds','Number of bathrooms','Amenities shown','Unavailable amenities shown','Number of amenities','Rating and Number of Reviews','Rating Breakdown']

# Reassign the DataFrame with columns in the desired order
result_df = result_df[new_order]

''' Split the 'Rating and Number of reviews' column into 'Overall rating' and 'Number of reviews'. '''

# Split the 'Rating and Number of reviews' into two columns
split_columns = result_df['Rating and Number of Reviews'].apply(pd.Series)

# Rename the new columns
split_columns = split_columns.rename(columns={0: 'Overall rating', 1: 'Number of reviews'})

# Concatenate the original DataFrame with the new columns
result_df2 = pd.concat([result_df, split_columns], axis=1)

''' For consistency, convert floats to integers and all missing values to 'N/A'. Note that ratings are
allowed to be floats. '''

# Function to convert floats to integers and replace NaN with 'N/A'
def convert_to_int_or_na(value):
    if pd.notna(value):
        return int(value)
    else:
        return 'N/A'

# Apply the custom function to the column
result_df2['Number of reviews'] = result_df2['Number of reviews'].apply(convert_to_int_or_na)

# Drop column 'Rating and Number of Reviews'
result_df2 = result_df2.drop('Rating and Number of Reviews', axis=1)

''' Split 'Rating Breakdown' columns into a column for each aspect that was given a rating. Note that ratings 
are allowed to be floats. '''

# Split the 'Rating Breakdown' into multiple columns
split_columns = result_df2['Rating Breakdown'].apply(pd.Series)

# Rename the new columns
split_columns = split_columns.rename(columns={0: 'Cleanliness', 1: 'Accuracy', 2: 'Check-in', 3: 'Communication', 4: 'Location', 5: 'Value'})

# Concatenate the original DataFrame with the new columns
result_df2 = pd.concat([result_df2, split_columns], axis=1)

# Drop column 'Rating Breakdown'
result_df2 = result_df2.drop('Rating Breakdown', axis=1)

''' From 'Summary' column, we extract the portion before 'hosted by...' because this is a description of
the type of dwelling advertised in the post. '''

def extract_before_hosted(data):
    # Define a regular expression pattern to match the text before 'hosted'
    pattern = r'^(.*?)\shosted'

    # Extract the desired part from each element in the list
    result = [re.search(pattern, item).group(1) if re.search(pattern, item) else item for item in data]

    return result

# Call the function and print the extracted results
result = extract_before_hosted(result_df2['Summary'])

# Update column 'Summary'
result_df2['Summary'] = result

''' Clean text in 'Title' column. '''

nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")
nltk.download("wordnet")
nltk.download("stopwords")
nltk.download('omw-1.4')

''' The following three functions are adopted from the 'text_classification_glove_embeddings.ipynb' notebook 
by embedded-robotics (https://github.com/embedded-robotics/datascience)'''

def get_wordnet_pos(tag):
    ''' Map part-of-speech tags to WordNet POS tags for lemmatization. This function converts Penn Treebank 
    POS tags (e.g., 'JJ', 'VB', 'NN', 'RB') to WordNet POS tags. '''
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def lemmatize(word_list):
    ''' Lemmatize a list of words using WordNetLemmatizer. This function takes a list of words, 
    performs part-of-speech tagging, and then lemmatizes each word based on its POS tag using WordNetLemmatizer.
     It returns the lemmatized words as a space-separated string. '''
    wl = WordNetLemmatizer()
    word_pos_tags = pos_tag(word_list)
    lemmatized_list = []
    for tag in word_pos_tags:
        lemmatize_word = wl.lemmatize(tag[0], get_wordnet_pos(tag[1]))
        lemmatized_list.append(lemmatize_word)
    return " ".join(lemmatized_list)

def clean_text(text):
    ''' Clean and preprocess the input text.'''
    text = str(text).strip()
    text = str(text).lower()
    text = re.sub(r"\n", r" ", text)
    word_tokens = word_tokenize(text)
    cleaned_text = []
    for word in word_tokens:
        cleaned_text.append("".join([char for char in word if char.isalnum()]))
    stop_words = stopwords.words('english')
    text_tokens = [word for word in cleaned_text if (len(word) > 2) and (word not in stop_words)]
    text = lemmatize(text_tokens)
    return text

# Apply text cleaning function to comments
title_col = result_df2['Title']
title_clean = title_col.apply(lambda x: clean_text(x))

# Store comments as lists of words
title_clean_list = title_clean.str.split(' ')

# Store text as one list of words
single_list = [item for sublist in title_clean_list for item in sublist]

''' From the titles of the posts, we want to see the most common words used. These words mostly describe the
charm and location of the properties. '''

# Frequency Distribution
fdist = FreqDist(single_list)

# Print most common words
print("Most common words:")
for word, frequency in fdist.most_common(100):
    ''' Here we print the 30 most common words '''
    print(f"{word}: {frequency}")

# Create a list of the top 100 most common words in titles
title_keywords = [word for word, _ in fdist.most_common(100)]

# Display keyword from 'Title' columns
print(title_keywords)

# Remove unnecessary keywords
values_to_remove = ['toronto','']

# Remove the specified values from the list
my_list = [x for x in title_keywords if x not in values_to_remove]

# Display list
print(my_list)

# Add a new column 'Title clean'
result_df2['Title clean'] = title_clean_list

# Create a new DataFrame with columns for each word
for word in my_list:
    result_df2[word] = result_df2['Title clean'].apply(lambda x: 1 if word in x else 0)

''' Explore the 30 most frequent keywords used in the titles of the posts. '''

# Bar plot of word frequency
word_freq_df = pd.DataFrame(fdist.most_common(30), columns=['Word', 'Frequency'])
plt.figure(figsize=(10, 6))
sns.barplot(x='Frequency', y='Word', data=word_freq_df)
plt.title('Top 30 Most Common Words in Title')
plt.show()

# Store text as one list of words
single_list = [item for sublist in result_df2['Amenities shown'] for item in sublist]

# Drop 'N', '/', 'A' elements
filtered_list = [item for item in single_list if item not in ('N', '/', 'A')]

# Frequency Distribution
fdist = FreqDist(filtered_list)

''' Explore the 30 frequently shown amenities (on the main page of the post). Split each amenity into its
own column and if the post shows that amenity assign '1', otherwise assign '0'. '''

# Bar plot of word frequency
word_freq_df = pd.DataFrame(fdist.most_common(30), columns=['Word', 'Frequency'])
plt.figure(figsize=(10, 6))
sns.barplot(x='Frequency', y='Word', data=word_freq_df)
plt.title('Top 30 Most Common Amenities')
plt.show()

# Print most common words
print("Most common words:")
for word, frequency in fdist.most_common(101):
    ''' Here we print the 101 most common words '''
    print(f"{word}: {frequency}")

amenities_keywords = [word for word, _ in fdist.most_common(101)]

# Create a new DataFrame with columns for each word
for word in amenities_keywords:
    result_df2[word] = result_df2['Amenities shown'].apply(lambda x: 1 if word in x else 0)

# Drop the original 'Amenities shown' column (optional)
result_df2.drop('Amenities shown', axis=1, inplace=True)

''' Explore the frequency of types of dwellings in the posts. '''

# Filter out 'N/A' values from the 'Summary' column
filtered_summary = result_df2['Summary'][result_df2['Summary'] != 'N/A']

# Create a histogram of the filtered titles
filtered_summary.value_counts().plot(kind='bar')

# Set labels and title for the histogram
plt.ylabel('Frequency')
plt.title('Histogram of Airbnb Descriptions')

# Show the histogram
plt.show()

''' Export data set '''

result_df2.to_csv('C:/Users/lynds/OneDrive/Documents/Python_Projects/Airbnb_CleanData.csv', index=False)