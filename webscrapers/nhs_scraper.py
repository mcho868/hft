import requests
from bs4 import BeautifulSoup
import time
import json
import os
from urllib.parse import urljoin
import argparse
import re
import random

# --- Configuration ---
HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
MIN_DELAY = 1
MAX_DELAY = 3
NHS_DATA_DIR = 'nhs_data'
NHS_URL_LIST_FILE = 'nhs_urls.json'
NHS_EXISTING_URLS_FILE = 'nhs_existing_urls.json'

def sanitize_filename(filename):
    """Removes invalid characters from a string to make it a valid filename."""
    return re.sub(r'[\\/*?:"<>|]', "", filename).replace(" ", "_")

def load_existing_urls():
    """Loads the set of already scraped URLs from the nhs_existing_urls.json file."""
    if os.path.exists(NHS_EXISTING_URLS_FILE):
        print(f"Loading existing URLs from '{NHS_EXISTING_URLS_FILE}'...")
        try:
            with open(NHS_EXISTING_URLS_FILE, 'r', encoding='utf-8') as f:
                existing_urls_list = json.load(f)
            existing_urls = set(existing_urls_list)
            print(f"Loaded {len(existing_urls)} existing URLs to skip.")
            return existing_urls
        except Exception as e:
            print(f"Error loading existing URLs: {e}")
            return set()
    else:
        print(f"No existing URLs file found at '{NHS_EXISTING_URLS_FILE}'. Will not skip any URLs.")
        return set()

def save_existing_urls(existing_urls):
    """Saves the set of scraped URLs back to the nhs_existing_urls.json file."""
    try:
        existing_urls_list = list(existing_urls)
        with open(NHS_EXISTING_URLS_FILE, 'w', encoding='utf-8') as f:
            json.dump(existing_urls_list, f, indent=4, ensure_ascii=False)
        print(f"Updated existing URLs file with {len(existing_urls)} URLs.")
    except Exception as e:
        print(f"Error saving existing URLs: {e}")

def discover_all_nhs_conditions():
    """Discovers all NHS condition URLs from the main conditions page."""
    base_url = 'https://www.nhs.uk'
    conditions_url = 'https://www.nhs.uk/conditions/'
    
    print(f"Discovering all NHS conditions from {conditions_url}...")
    
    try:
        response = requests.get(conditions_url, headers=HEADERS)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find all condition links in the nhsuk-list
        condition_urls = set()
        for link in soup.select('ol.nhsuk-list a[href^="/conditions/"]'):
            href = link.get('href')
            if href:
                # Skip anchor links like /conditions/#a, #b, etc.
                if '#' in href and href.count('/') == 1:
                    continue
                full_url = urljoin(base_url, href)
                condition_urls.add(full_url)
        
        condition_urls_list = sorted(list(condition_urls))
        print(f"Discovered {len(condition_urls_list)} NHS condition URLs.")
        
        # Save the discovered URLs
        with open(NHS_URL_LIST_FILE, 'w', encoding='utf-8') as f:
            json.dump(condition_urls_list, f, indent=4, ensure_ascii=False)
        
        return condition_urls_list
        
    except Exception as e:
        print(f"Error discovering NHS conditions: {e}")
        return []

def parse_nhs_content(html, url):
    """Parses NHS condition page content and extracts structured text."""
    soup = BeautifulSoup(html, 'html.parser')
    
    # Get the main title
    title_element = soup.select_one('h1')
    title = title_element.get_text(strip=True) if title_element else "Unknown Condition"
    
    # Find the main content area
    main_content = soup.select_one('main[id="maincontent"]')
    if not main_content:
        main_content = soup.select_one('article')
    if not main_content:
        main_content = soup
    
    # Extract structured content by sections
    sections = []
    current_section = None
    section_content = []
    
    # Look for content elements in order
    content_elements = main_content.find_all(['h2', 'h3', 'p', 'ul', 'li', 'div'])
    
    for element in content_elements:
        # Skip navigation and other non-content elements
        if element.get('class'):
            classes = ' '.join(element.get('class', []))
            if any(skip in classes for skip in ['nhsuk-breadcrumb', 'nhsuk-skip-link', 'nhsuk-header', 'nhsuk-footer']):
                continue
        
        if element.name in ['h2', 'h3']:
            # Save previous section if it exists
            if current_section and section_content:
                sections.append(f"--- {current_section} ---")
                sections.extend(section_content)
                sections.append("")
            
            # Start new section
            current_section = element.get_text(strip=True)
            section_content = []
            
        elif element.name == 'p':
            text = element.get_text(strip=True)
            if text and len(text) > 10:  # Filter out very short paragraphs
                section_content.append(text)
                
        elif element.name == 'li':
            text = element.get_text(strip=True)
            if text:
                section_content.append(f"• {text}")
                
        elif element.name == 'ul':
            # Process list items within ul
            for li in element.find_all('li', recursive=False):
                text = li.get_text(strip=True)
                if text:
                    section_content.append(f"• {text}")
    
    # Add the last section
    if current_section and section_content:
        sections.append(f"--- {current_section} ---")
        sections.extend(section_content)
    
    # If no sections found, try to get main content directly
    if not sections:
        main_text_elements = main_content.find_all(['p', 'li'])
        for element in main_text_elements:
            text = element.get_text(strip=True)
            if text and len(text) > 10:
                if element.name == 'li':
                    sections.append(f"• {text}")
                else:
                    sections.append(text)
    
    content_text = '\n\n'.join(sections) if sections else None
    return title, content_text

def save_nhs_condition_data(condition_data):
    """Saves a single NHS condition's data to its own JSON file."""
    if not os.path.exists(NHS_DATA_DIR):
        os.makedirs(NHS_DATA_DIR)
    
    title = condition_data.get("title", "Unknown_Condition")
    filename = sanitize_filename(title) + ".json"
    filepath = os.path.join(NHS_DATA_DIR, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(condition_data, f, indent=4, ensure_ascii=False)
    print(f"  -> Successfully saved condition to '{filepath}'")

def crawl_nhs_conditions(is_test_mode=False):
    """
    Crawls NHS conditions with resumability and duplicate prevention.
    """
    print("\n--- Starting NHS conditions scraper ---")
    
    # Load existing URLs to skip duplicates
    existing_urls = load_existing_urls()
    
    # Discover or load condition URLs
    condition_urls = []
    if os.path.exists(NHS_URL_LIST_FILE):
        print(f"Loading condition URLs from '{NHS_URL_LIST_FILE}'...")
        with open(NHS_URL_LIST_FILE, 'r', encoding='utf-8') as f:
            condition_urls = json.load(f)
        print(f"Loaded {len(condition_urls)} condition URLs.")
    else:
        print("No URL list found. Discovering NHS condition URLs...")
        condition_urls = discover_all_nhs_conditions()
    
    if not condition_urls:
        print("No condition URLs found. Exiting.")
        return
    
    # Limit URLs for test mode
    urls_to_scrape = condition_urls[:3] if is_test_mode else condition_urls
    
    print(f"\nStarting to scrape {len(urls_to_scrape)} conditions...")
    
    for i, condition_url in enumerate(urls_to_scrape):
        print(f"\n--- Processing condition {i+1}/{len(urls_to_scrape)}: {condition_url} ---")
        
        # Skip if already processed
        if condition_url in existing_urls:
            print(f"  -> URL already scraped. Skipping.")
            continue
        
        try:
            # Add delay between requests
            time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))
            
            # Fetch the condition page
            response = requests.get(condition_url, headers=HEADERS, timeout=30)
            response.raise_for_status()
            
            # Parse the content
            title, content = parse_nhs_content(response.text, condition_url)
            
            if content and content.strip():
                # Save the condition data
                condition_data = {
                    'source': 'nhs',
                    'url': condition_url,
                    'title': title,
                    'text': content
                }
                save_nhs_condition_data(condition_data)
                
                # Add to existing URLs and save immediately
                existing_urls.add(condition_url)
                save_existing_urls(existing_urls)
                
                print(f"  -> Successfully processed '{title}'")
            else:
                print(f"  -> No content found for '{title}', skipping.")
                
        except Exception as e:
            print(f"  -> Error processing {condition_url}: {e}")
            continue
    
    print(f"\nCompleted NHS scraping. Processed {len(existing_urls)} total conditions.")

def main():
    parser = argparse.ArgumentParser(description="Scrape NHS conditions.")
    parser.add_argument('--test', action='store_true', help='Run in test mode (scrape only 3 conditions).')
    args = parser.parse_args()

    global MIN_DELAY, MAX_DELAY
    if args.test:
        print("\n--- RUNNING IN TEST MODE ---")
        MIN_DELAY = 0.5
        MAX_DELAY = 1.0

    crawl_nhs_conditions(is_test_mode=args.test)

if __name__ == '__main__':
    main()