import requests
from bs4 import BeautifulSoup
import time
import json
import os
from urllib.parse import urljoin, urlparse
import argparse
import re
import random # Added for randomized delays

# --- Selenium Imports (for advanced scraping with login) ---
# Selenium is needed for websites that require login or have dynamic JavaScript content.
try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from webdriver_manager.chrome import ChromeDriverManager
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False

# --- Configuration ---
HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
MIN_DELAY = 2
MAX_DELAY = 5
# --- UPDATED: New configuration for resilient scraping ---
MEDSCAPE_DATA_DIR = 'medscape_data'
MEDSCAPE_URL_LIST_FILE = 'medscape_topic_urls.json'
EXISTING_URLS_FILE = 'existing_urls.json'


# --- Medscape Specific Functions (using Selenium) ---

def setup_driver():
    """Initializes the Selenium WebDriver."""
    print("Setting up Selenium WebDriver...")
    options = webdriver.ChromeOptions()
    # The browser window will be visible for manual login
    # options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument(f'user-agent={HEADERS["User-Agent"]}')
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    return driver

def login_to_medscape_manual(driver):
    """Pauses for manual login and trusts the user."""
    print("\n--- MANUAL LOGIN REQUIRED ---")
    login_url = 'https://login.medscape.com/login/sso/getlogin?ac=401'
    driver.get(login_url)
    input("A browser window has opened. Please log in to Medscape manually, then press Enter here to continue...")
    print("Resuming script. Assuming login was successful.")
    return True

def parse_medscape_content_structured(html):
    """
    Parses Medscape content by first finding a valid content container
    (either 'content_a...' or 'drugdbmain') and then extracting structured text.
    """
    soup = BeautifulSoup(html, 'html.parser')
    title = soup.select_one('h1.topic-title, h1.article-title, div#maincolboxdrugdbheader h1')
    title_text = title.get_text(strip=True) if title else "No Title Found"
    
    content_section = soup.select_one('div[id^="content_a"]')
    if not content_section:
        content_section = soup.select_one('div#drugdbmain')

    if not content_section:
        return title_text, None

    allowed_tags = ['h2', 'h3', 'p', 'ul', 'li']
    elements = content_section.find_all(allowed_tags)
    
    structured_text = []
    for element in elements:
        if element.name == 'li':
            text = element.get_text(strip=True)
            if text: structured_text.append(f"- {text}")
        else:
            text = element.get_text(strip=True)
            if text: structured_text.append(text)
            
    content = '\n\n'.join(filter(None, structured_text))
    return title_text, content if content.strip() else None

def sanitize_filename(filename):
    """Removes invalid characters from a string to make it a valid filename."""
    return re.sub(r'[\\/*?:"<>|]', "", filename).replace(" ", "_")

def load_existing_urls():
    """Loads the set of already scraped URLs from the existing_urls.json file."""
    if os.path.exists(EXISTING_URLS_FILE):
        print(f"Loading existing URLs from '{EXISTING_URLS_FILE}'...")
        try:
            with open(EXISTING_URLS_FILE, 'r', encoding='utf-8') as f:
                existing_urls_list = json.load(f)
            existing_urls = set(existing_urls_list)
            print(f"Loaded {len(existing_urls)} existing URLs to skip.")
            return existing_urls
        except Exception as e:
            print(f"Error loading existing URLs: {e}")
            return set()
    else:
        print(f"No existing URLs file found at '{EXISTING_URLS_FILE}'. Will not skip any URLs.")
        return set()

def save_existing_urls(existing_urls):
    """Saves the set of scraped URLs back to the existing_urls.json file."""
    try:
        existing_urls_list = list(existing_urls)
        with open(EXISTING_URLS_FILE, 'w', encoding='utf-8') as f:
            json.dump(existing_urls_list, f, indent=4, ensure_ascii=False)
        print(f"Updated existing URLs file with {len(existing_urls)} URLs.")
    except Exception as e:
        print(f"Error saving existing URLs: {e}")

def save_topic_data(topic_data):
    """Saves a single topic's data to its own JSON file."""
    if not os.path.exists(MEDSCAPE_DATA_DIR):
        os.makedirs(MEDSCAPE_DATA_DIR)
    
    title = topic_data.get("title", "Untitled_Topic")
    filename = sanitize_filename(title) + ".json"
    filepath = os.path.join(MEDSCAPE_DATA_DIR, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(topic_data, f, indent=4, ensure_ascii=False)
    print(f"  -> Successfully saved topic to '{filepath}'")

def crawl_medscape(driver, is_test_mode=False):
    """
    Final robust version: Discovers all topics, saves the list, then scrapes
    each one, saving progress individually.
    """
    print("\n--- Starting resilient crawl for Medscape ---")
    base_url = 'https://emedicine.medscape.com'
    
    # Load existing URLs to skip duplicates
    existing_urls = load_existing_urls()
    
    # --- Level 1 & 2: Discover all topic URLs (or load from file) ---
    base_article_urls = []
    if os.path.exists(MEDSCAPE_URL_LIST_FILE):
        print(f"Found existing URL list '{MEDSCAPE_URL_LIST_FILE}', loading URLs...")
        with open(MEDSCAPE_URL_LIST_FILE, 'r') as f:
            base_article_urls = json.load(f)
    else:
        print("No URL list found. Discovering all topic URLs now (this will take a while)...")
        print("Level 1: Finding specialties...")
        driver.get(base_url + '/')
        time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        specialty_links = {urljoin(base_url, a['href']) for a in soup.select('div.browse-medicine a, div.browse-surgery a') if a.get('href')}
        print(f"Found {len(specialty_links)} specialties.")

        print("\nLevel 2: Finding base article topics...")
        discovered_urls = set()
        for i, specialty_url in enumerate(sorted(list(specialty_links))):
            print(f"  -> Processing specialty {i+1}/{len(specialty_links)}: {specialty_url}")
            driver.get(specialty_url)
            time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            for a in soup.select('div.topic-list a'):
                href = a.get('href')
                if href and 'article' in href:
                    discovered_urls.add(urljoin(base_url, href.split('-')[0]))
        
        base_article_urls = sorted(list(discovered_urls))
        with open(MEDSCAPE_URL_LIST_FILE, 'w') as f:
            json.dump(base_article_urls, f, indent=4)
        print(f"\nDiscovered and saved {len(base_article_urls)} unique medical topics to '{MEDSCAPE_URL_LIST_FILE}'.")

    # --- Level 3: Scrape all sub-pages for each topic ---
    print(f"\nLevel 3: Beginning scrape of {len(base_article_urls)} topics...")
    sub_pages = ['overview', 'clinical', 'differential', 'workup', 'treatment', 'guidelines', 'medication']
    
    topics_to_scrape = base_article_urls[:2] if is_test_mode else base_article_urls

    for i, base_topic_url in enumerate(topics_to_scrape):
        print(f"\n--- Processing Topic {i+1}/{len(topics_to_scrape)} (Base: {base_topic_url}) ---")

        # --- Progress Check - Skip if URL already scraped ---
        if base_topic_url in existing_urls:
            print(f"  -> URL '{base_topic_url}' already scraped. Skipping.")
            continue
        
        full_topic_text = []
        main_title = "No Title Found"
        
        for sub_page_name in sub_pages:
            full_content_url = f"{base_topic_url}-{sub_page_name}#showall"
            print(f"  -> Fetching sub-page: {full_content_url}")
            driver.get(full_content_url)
            
            try:
                WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, 'body')))
            except Exception:
                print(f"     Sub-page did not exist or failed to load, skipping.")
                continue
            
            title, content = parse_medscape_content_structured(driver.page_source)
            if content:
                print(f"     Successfully parsed content for '{sub_page_name}'.")
                if sub_page_name == 'overview' and title != "No Title Found":
                    main_title = title
                
                full_topic_text.append(f"--- Section: {sub_page_name.capitalize()} ---")
                full_topic_text.append(content)
            else:
                 print(f"     No valid content found for '{sub_page_name}', skipping.")
        
        if full_topic_text:
            topic_data = {
                'source': 'medscape',
                'url': base_topic_url,
                'title': main_title,
                'text': '\n\n'.join(full_topic_text)
            }
            save_topic_data(topic_data)
            
            # Add URL to existing_urls and save to prevent future duplicates
            existing_urls.add(base_topic_url)
            save_existing_urls(existing_urls)


def main():
    parser = argparse.ArgumentParser(description="Scrape medical websites.")
    parser.add_argument('--test', action='store_true', help='Run in test mode.')
    args = parser.parse_args()

    global MIN_DELAY, MAX_DELAY
    if args.test:
        print("\n--- RUNNING IN TEST MODE ---")
        MIN_DELAY = 0.5
        MAX_DELAY = 1.0

    # Run Medscape scraper if Selenium is available
    if SELENIUM_AVAILABLE:
        driver = setup_driver()
        try:
            if login_to_medscape_manual(driver):
                crawl_medscape(driver, is_test_mode=args.test)
        finally:
            print("Closing WebDriver.")
            driver.quit()
    else:
        print("\nSelenium is not installed. Skipping Medscape.")
        print("Please run 'pip install selenium webdriver-manager' to scrape Medscape.")

if __name__ == '__main__':
    main()

#mcho868@aucklanduni.ac.nz
#TyDc67pnbBhV#!@