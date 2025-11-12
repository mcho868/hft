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
MAYO_DATA_DIR = 'mayo_data'
MAYO_URL_LIST_FILE = 'mayo_urls.json'
MAYO_EXISTING_URLS_FILE = 'mayo_existing_urls.json'
DIAGNOSIS_TREATMENT_EXISTING_URLS_FILE = 'mayo_diagnosis_treatment_existing_urls.json'

def sanitize_filename(filename):
    """Removes invalid characters from a string to make it a valid filename."""
    return re.sub(r'[\\/*?:"<>|]', "", filename).replace(" ", "_")

def load_existing_urls():
    """Loads the set of already scraped diagnosis-treatment URLs."""
    if os.path.exists(DIAGNOSIS_TREATMENT_EXISTING_URLS_FILE):
        print(f"Loading existing diagnosis-treatment URLs from '{DIAGNOSIS_TREATMENT_EXISTING_URLS_FILE}'...")
        try:
            with open(DIAGNOSIS_TREATMENT_EXISTING_URLS_FILE, 'r', encoding='utf-8') as f:
                existing_urls_list = json.load(f)
            existing_urls = set(existing_urls_list)
            print(f"Loaded {len(existing_urls)} existing diagnosis-treatment URLs to skip.")
            return existing_urls
        except Exception as e:
            print(f"Error loading existing URLs: {e}")
            return set()
    else:
        print(f"No existing diagnosis-treatment URLs file found. Will not skip any URLs.")
        return set()

def save_existing_urls(existing_urls):
    """Saves the set of scraped diagnosis-treatment URLs."""
    try:
        existing_urls_list = list(existing_urls)
        with open(DIAGNOSIS_TREATMENT_EXISTING_URLS_FILE, 'w', encoding='utf-8') as f:
            json.dump(existing_urls_list, f, indent=4, ensure_ascii=False)
        print(f"Updated existing diagnosis-treatment URLs file with {len(existing_urls)} URLs.")
    except Exception as e:
        print(f"Error saving existing URLs: {e}")

def find_diagnosis_treatment_url(symptoms_causes_url):
    """Finds the diagnosis-treatment URL from a symptoms-causes page by looking for navigation links."""
    try:
        print(f"  -> Finding diagnosis-treatment URL from: {symptoms_causes_url}")
        
        time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))
        response = requests.get(symptoms_causes_url, headers=HEADERS, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Look for navigation menu items with role="menuitem" containing diagnosis-treatment links
        menu_items = soup.select('li[role="menuitem"]')
        
        for menu_item in menu_items:
            link = menu_item.select_one('a[href*="diagnosis-treatment"]')
            if link:
                href = link.get('href')
                if href:
                    # Convert relative URL to absolute
                    full_url = urljoin('https://www.mayoclinic.org', href)
                    print(f"    Found diagnosis-treatment URL: {full_url}")
                    return full_url
        
        # Fallback: look for any diagnosis-treatment link in the page
        diagnosis_links = soup.select('a[href*="diagnosis-treatment"]')
        for link in diagnosis_links:
            href = link.get('href')
            if href and '/diseases-conditions/' in href:
                full_url = urljoin('https://www.mayoclinic.org', href)
                print(f"    Found diagnosis-treatment URL (fallback): {full_url}")
                return full_url
        
        print(f"    No diagnosis-treatment URL found for {symptoms_causes_url}")
        return None
        
    except Exception as e:
        print(f"    Error finding diagnosis-treatment URL for {symptoms_causes_url}: {e}")
        return None

def parse_mayo_content(html):
    """Parses Mayo Clinic diagnosis-treatment page content and extracts structured text."""
    soup = BeautifulSoup(html, 'html.parser')
    
    # Get the main title
    title_element = soup.select_one('h1')
    title = title_element.get_text(strip=True) if title_element else "Unknown Condition"
    
    # Remove navigation, header, footer, and sidebar elements
    for element in soup.select('nav, header, footer, .breadcrumb, .site-header, .site-footer, .sidebar, .navigation, .nav, #header, #footer, #sidebar'):
        element.decompose()
    
    # Remove Mayo Clinic promotional content and links
    for element in soup.select('aside, .ad, .advertisement, .promo, .related-links, .book-promo, .newsletter-signup'):
        element.decompose()
    
    # Find the main content area
    main_content = soup.select_one('article[id="main-content"], .main-content, main, [role="main"]')
    if not main_content:
        main_content = soup.select_one('div[class*="content"]')
    if not main_content:
        main_content = soup
    
    # Look for content elements in order, but skip common non-content areas
    content_elements = main_content.find_all(['h2', 'h3', 'p', 'ul', 'li'])
    
    # Filter out elements that are likely navigation or promotional content
    filtered_elements = []
    for element in content_elements:
        text = element.get_text(strip=True).lower()
        
        # Skip elements that contain common non-medical navigation terms
        skip_phrases = [
            'mayo clinic', 'request appointment', 'find a doctor', 'patient portal',
            'giving to mayo', 'contact us', 'locations', 'careers', 'sign up',
            'newsletter', 'book', 'advertisement', 'privacy policy', 'terms',
            'follow mayo clinic', 'get the app', 'site map', 'manage cookies'
        ]
        
        if any(phrase in text for phrase in skip_phrases):
            continue
        
        # Skip very short elements or elements that are just links
        if len(text) < 10 or (element.name in ['p', 'li'] and len(element.find_all('a')) > len(text.split()) // 2):
            continue
            
        filtered_elements.append(element)
    
    content_elements = filtered_elements
    
    # Extract structured content by sections
    sections = []
    current_section = None
    section_content = []
    
    for element in content_elements:
        # Skip navigation and other non-content elements
        if element.get('class'):
            classes = ' '.join(element.get('class', []))
            if any(skip in classes for skip in ['breadcrumb', 'navigation', 'sidebar', 'footer', 'header']):
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
            if text and len(text) > 15:  # Filter out very short paragraphs
                section_content.append(text)
                
        elif element.name == 'li':
            text = element.get_text(strip=True)
            if text and len(text) > 5:
                section_content.append(f"• {text}")
                
        elif element.name == 'ul':
            # Process list items within ul
            for li in element.find_all('li', recursive=False):
                text = li.get_text(strip=True)
                if text and len(text) > 5:
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
            if text and len(text) > 15:
                if element.name == 'li':
                    sections.append(f"• {text}")
                else:
                    sections.append(text)
    
    content_text = '\n\n'.join(sections) if sections else None
    return title, content_text

def save_diagnosis_treatment_data(condition_data):
    """Saves a single Mayo condition's diagnosis-treatment data to its own JSON file."""
    if not os.path.exists(MAYO_DATA_DIR):
        os.makedirs(MAYO_DATA_DIR)
    
    title = condition_data.get("title", "Unknown_Condition")
    filename = sanitize_filename(title) + "_diagnosis_treatment.json"
    filepath = os.path.join(MAYO_DATA_DIR, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(condition_data, f, indent=4, ensure_ascii=False)
    print(f"  -> Successfully saved diagnosis-treatment data to '{filepath}'")

def crawl_mayo_diagnosis_treatment(is_test_mode=False):
    """
    Crawls Mayo Clinic diagnosis-treatment pages based on existing symptoms-causes URLs.
    """
    print("\n--- Starting Mayo Clinic diagnosis-treatment scraper ---")
    
    # Load existing diagnosis-treatment URLs to skip duplicates
    existing_urls = load_existing_urls()
    
    # Load symptoms-causes URLs from the main URL list
    symptoms_causes_urls = []
    if os.path.exists(MAYO_URL_LIST_FILE):
        print(f"Loading symptoms-causes URLs from '{MAYO_URL_LIST_FILE}'...")
        with open(MAYO_URL_LIST_FILE, 'r', encoding='utf-8') as f:
            symptoms_causes_urls = json.load(f)
        print(f"Loaded {len(symptoms_causes_urls)} symptoms-causes URLs.")
    else:
        print(f"No URL list found at '{MAYO_URL_LIST_FILE}'. Exiting.")
        return
    
    if not symptoms_causes_urls:
        print("No symptoms-causes URLs found. Exiting.")
        return
    
    # Limit URLs for test mode
    urls_to_process = symptoms_causes_urls[:2] if is_test_mode else symptoms_causes_urls
    
    print(f"\nStarting to process {len(urls_to_process)} symptoms-causes URLs to find diagnosis-treatment pages...")
    
    for i, symptoms_causes_url in enumerate(urls_to_process):
        print(f"\n--- Processing {i+1}/{len(urls_to_process)}: {symptoms_causes_url} ---")
        
        try:
            # Find the diagnosis-treatment URL
            diagnosis_treatment_url = find_diagnosis_treatment_url(symptoms_causes_url)
            
            if not diagnosis_treatment_url:
                print(f"  -> No diagnosis-treatment URL found, skipping.")
                continue
            
            # Skip if already processed
            if diagnosis_treatment_url in existing_urls:
                print(f"  -> Diagnosis-treatment URL already scraped. Skipping.")
                continue
            
            # Fetch and parse the diagnosis-treatment content
            print(f"  -> Fetching diagnosis-treatment content: {diagnosis_treatment_url}")
            
            time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))
            response = requests.get(diagnosis_treatment_url, headers=HEADERS, timeout=30)
            response.raise_for_status()
            
            title, content = parse_mayo_content(response.text)
            
            if content and content.strip():
                condition_data = {
                    'source': 'mayo',
                    'url': diagnosis_treatment_url,
                    'title': title,
                    'text': f"--- Diagnosis Treatment Section ---\n\n{content}"
                }
                save_diagnosis_treatment_data(condition_data)
                
                # Add to existing URLs and save immediately
                existing_urls.add(diagnosis_treatment_url)
                save_existing_urls(existing_urls)
                
                print(f"  -> Successfully processed '{title}' (Diagnosis Treatment)")
            else:
                print(f"  -> No content found for {diagnosis_treatment_url}")
                
        except Exception as e:
            print(f"  -> Error processing {symptoms_causes_url}: {e}")
            continue
    
    print(f"\nCompleted Mayo Clinic diagnosis-treatment scraping. Processed {len(existing_urls)} total diagnosis-treatment conditions.")

def main():
    parser = argparse.ArgumentParser(description="Scrape Mayo Clinic diagnosis-treatment pages.")
    parser.add_argument('--test', action='store_true', help='Run in test mode (process only 2 URLs).')
    args = parser.parse_args()

    global MIN_DELAY, MAX_DELAY
    if args.test:
        print("\n--- RUNNING IN TEST MODE ---")
        MIN_DELAY = 0.5
        MAX_DELAY = 1.0

    crawl_mayo_diagnosis_treatment(is_test_mode=args.test)

if __name__ == '__main__':
    main()