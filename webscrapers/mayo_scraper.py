import requests
from bs4 import BeautifulSoup
import time
import json
import os
from urllib.parse import urljoin, urlparse
import argparse
import re
import random
import string

# --- Configuration ---
HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
MIN_DELAY = 1
MAX_DELAY = 3
MAYO_DATA_DIR = 'mayo_data'
MAYO_URL_LIST_FILE = 'mayo_urls.json'
MAYO_EXISTING_URLS_FILE = 'mayo_existing_urls.json'

def sanitize_filename(filename):
    """Removes invalid characters from a string to make it a valid filename."""
    return re.sub(r'[\\/*?:"<>|]', "", filename).replace(" ", "_")

def load_existing_urls():
    """Loads the set of already scraped URLs from the mayo_existing_urls.json file."""
    if os.path.exists(MAYO_EXISTING_URLS_FILE):
        print(f"Loading existing URLs from '{MAYO_EXISTING_URLS_FILE}'...")
        try:
            with open(MAYO_EXISTING_URLS_FILE, 'r', encoding='utf-8') as f:
                existing_urls_list = json.load(f)
            existing_urls = set(existing_urls_list)
            print(f"Loaded {len(existing_urls)} existing URLs to skip.")
            return existing_urls
        except Exception as e:
            print(f"Error loading existing URLs: {e}")
            return set()
    else:
        print(f"No existing URLs file found at '{MAYO_EXISTING_URLS_FILE}'. Will not skip any URLs.")
        return set()

def save_existing_urls(existing_urls):
    """Saves the set of scraped URLs back to the mayo_existing_urls.json file."""
    try:
        existing_urls_list = list(existing_urls)
        with open(MAYO_EXISTING_URLS_FILE, 'w', encoding='utf-8') as f:
            json.dump(existing_urls_list, f, indent=4, ensure_ascii=False)
        print(f"Updated existing URLs file with {len(existing_urls)} URLs.")
    except Exception as e:
        print(f"Error saving existing URLs: {e}")

def discover_mayo_conditions_from_letter(letter):
    """Discovers condition URLs from a specific letter page."""
    base_url = 'https://www.mayoclinic.org'
    letter_url = f'https://www.mayoclinic.org/diseases-conditions/index?letter={letter.upper()}'
    
    print(f"  -> Discovering conditions for letter '{letter.upper()}': {letter_url}")
    
    try:
        time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))
        response = requests.get(letter_url, headers=HEADERS, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find condition links from the alphabetical listing using exact href attributes
        condition_urls = set()
        
        # Look for the main content area or condition list
        main_content = soup.select_one('#main-content, main, .content, .conditions-list')
        if not main_content:
            main_content = soup
        
        # Find all links that go to condition pages - use exact href values
        condition_links = main_content.select('a[href*="/diseases-conditions/"]')
        
        for link in condition_links:
            href = link.get('href')
            if not href:
                continue
                
            # Skip index pages, navigation links, and other non-condition pages
            if any(skip in href for skip in ['index', '?letter=', '/ar/', '/es/', '/zh-']):
                continue
                
            # Skip if it's just the base diseases-conditions path
            if href.endswith('/diseases-conditions/') or href == '/diseases-conditions':
                continue
            
            # Use the exact href attribute as found in the HTML
            # Don't modify the URL structure - keep it exactly as Mayo Clinic has it
            full_url = urljoin(base_url, href)
            
            # Filter out obvious navigation links but keep actual condition URLs
            # Accept both relative paths and full URLs
            if href.startswith('https://'):
                # This is already a full URL - check if it's a valid condition URL
                if '/diseases-conditions/' in href and not any(nav in href for nav in ['index', '?letter=']):
                    condition_urls.add(href)
            else:
                # This is a relative path - convert to full URL
                if '/diseases-conditions/' in href and not any(nav in href for nav in ['index', '?letter=']):
                    condition_urls.add(full_url)
        
        print(f"    Found {len(condition_urls)} conditions for letter {letter.upper()}")
        return list(condition_urls)
        
    except Exception as e:
        print(f"    Error discovering conditions for letter {letter}: {e}")
        return []

def discover_all_mayo_conditions():
    """Discovers all Mayo Clinic condition URLs by going through A-Z letters."""
    print("Discovering all Mayo Clinic conditions from A-Z letter pages...")
    
    all_condition_urls = set()
    
    # Go through each letter A-Z
    for letter in string.ascii_lowercase:
        letter_conditions = discover_mayo_conditions_from_letter(letter)
        all_condition_urls.update(letter_conditions)
    
    condition_urls_list = sorted(list(all_condition_urls))
    print(f"Discovered {len(condition_urls_list)} total Mayo Clinic condition URLs.")
    
    # Save the discovered URLs
    with open(MAYO_URL_LIST_FILE, 'w', encoding='utf-8') as f:
        json.dump(condition_urls_list, f, indent=4, ensure_ascii=False)
    
    return condition_urls_list

def find_condition_subpages(condition_url):
    """Finds the symptoms-causes and diagnosis-treatment subpages for a condition."""
    subpages = []
    
    try:
        time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))
        response = requests.get(condition_url, headers=HEADERS, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Look for navigation links to symptoms-causes and diagnosis-treatment
        base_url = 'https://www.mayoclinic.org'
        
        # Find symptoms-causes link
        symptoms_link = soup.select_one('a[href*="symptoms-causes"]')
        if symptoms_link:
            symptoms_url = urljoin(base_url, symptoms_link.get('href'))
            subpages.append(('symptoms-causes', symptoms_url))
        
        # Find diagnosis-treatment link  
        diagnosis_link = soup.select_one('a[href*="diagnosis-treatment"]')
        if diagnosis_link:
            diagnosis_url = urljoin(base_url, diagnosis_link.get('href'))
            subpages.append(('diagnosis-treatment', diagnosis_url))
            
        return subpages
        
    except Exception as e:
        print(f"    Error finding subpages for {condition_url}: {e}")
        return []

def parse_mayo_content(html, _):
    """Parses Mayo Clinic condition page content and extracts structured text."""
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
    
    # Extract structured content by sections
    sections = []
    current_section = None
    section_content = []
    
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

def save_mayo_condition_data(condition_data):
    """Saves a single Mayo condition's data to its own JSON file."""
    if not os.path.exists(MAYO_DATA_DIR):
        os.makedirs(MAYO_DATA_DIR)
    
    title = condition_data.get("title", "Unknown_Condition")
    filename = sanitize_filename(title) + ".json"
    filepath = os.path.join(MAYO_DATA_DIR, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(condition_data, f, indent=4, ensure_ascii=False)
    print(f"  -> Successfully saved condition to '{filepath}'")

def crawl_mayo_conditions(is_test_mode=False):
    """
    Crawls Mayo Clinic conditions with resumability and duplicate prevention.
    """
    print("\n--- Starting Mayo Clinic conditions scraper ---")
    
    # Load existing URLs to skip duplicates
    existing_urls = load_existing_urls()
    
    # Discover or load condition URLs
    condition_urls = []
    if os.path.exists(MAYO_URL_LIST_FILE):
        print(f"Loading condition URLs from '{MAYO_URL_LIST_FILE}'...")
        with open(MAYO_URL_LIST_FILE, 'r', encoding='utf-8') as f:
            condition_urls = json.load(f)
        print(f"Loaded {len(condition_urls)} condition URLs.")
    else:
        print("No URL list found. Discovering Mayo Clinic condition URLs...")
        condition_urls = discover_all_mayo_conditions()
    
    if not condition_urls:
        print("No condition URLs found. Exiting.")
        return
    
    # Limit URLs for test mode
    urls_to_scrape = condition_urls[:2] if is_test_mode else condition_urls
    
    print(f"\nStarting to scrape {len(urls_to_scrape)} conditions...")
    
    for i, condition_url in enumerate(urls_to_scrape):
        print(f"\n--- Processing condition {i+1}/{len(urls_to_scrape)}: {condition_url} ---")
        
        # Skip if already processed
        if condition_url in existing_urls:
            print(f"  -> URL already scraped. Skipping.")
            continue
        
        try:
            # Determine if this is already a specific subpage or if we need to find subpages
            if '/symptoms-causes/' in condition_url or '/diagnosis-treatment/' in condition_url:
                # This is already a specific subpage URL - scrape it directly
                print(f"  -> Fetching content from specific page: {condition_url}")
                
                time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))
                response = requests.get(condition_url, headers=HEADERS, timeout=30)
                response.raise_for_status()
                
                title, content = parse_mayo_content(response.text, condition_url)
                
                if content and content.strip():
                    # Determine section type from URL
                    if '/symptoms-causes/' in condition_url:
                        section_type = "Symptoms Causes Section"
                    elif '/diagnosis-treatment/' in condition_url:
                        section_type = "Diagnosis Treatment Section"
                    else:
                        section_type = "Overview Section"
                    
                    condition_data = {
                        'source': 'mayo',
                        'url': condition_url,
                        'title': title,
                        'text': f"--- {section_type} ---\n\n{content}"
                    }
                    save_mayo_condition_data(condition_data)
                    
                    # Add to existing URLs and save immediately
                    existing_urls.add(condition_url)
                    save_existing_urls(existing_urls)
                    
                    print(f"  -> Successfully processed '{title}' ({section_type})")
                else:
                    print(f"  -> No content found for {condition_url}")
            else:
                # This is a main condition page - find subpages (symptoms-causes and diagnosis-treatment)
                subpages = find_condition_subpages(condition_url)
                
                if not subpages:
                    print(f"  -> No subpages found for condition, skipping.")
                    continue
                
                # Combine content from all subpages
                full_content_sections = []
                main_title = "Unknown Condition"
                
                for section_type, subpage_url in subpages:
                    print(f"  -> Fetching {section_type}: {subpage_url}")
                    
                    time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))
                    response = requests.get(subpage_url, headers=HEADERS, timeout=30)
                    response.raise_for_status()
                    
                    title, content = parse_mayo_content(response.text, subpage_url)
                    
                    if content and content.strip():
                        if section_type == 'symptoms-causes' and title != "Unknown Condition":
                            main_title = title
                        
                        full_content_sections.append(f"--- {section_type.replace('-', ' ').title()} Section ---")
                        full_content_sections.append(content)
                        full_content_sections.append("")
                        
                        print(f"    Successfully parsed {section_type} content")
                    else:
                        print(f"    No content found for {section_type}")
                
                if full_content_sections:
                    # Save combined condition data
                    condition_data = {
                        'source': 'mayo',
                        'url': condition_url,
                        'title': main_title,
                        'text': '\n\n'.join(full_content_sections)
                    }
                    save_mayo_condition_data(condition_data)
                    
                    # Add to existing URLs and save immediately
                    existing_urls.add(condition_url)
                    save_existing_urls(existing_urls)
                    
                    print(f"  -> Successfully processed '{main_title}'")
                else:
                    print(f"  -> No content found for condition, skipping.")
                
        except Exception as e:
            print(f"  -> Error processing {condition_url}: {e}")
            continue
    
    print(f"\nCompleted Mayo Clinic scraping. Processed {len(existing_urls)} total conditions.")

def main():
    parser = argparse.ArgumentParser(description="Scrape Mayo Clinic conditions.")
    parser.add_argument('--test', action='store_true', help='Run in test mode (scrape only 2 conditions).')
    args = parser.parse_args()

    global MIN_DELAY, MAX_DELAY
    if args.test:
        print("\n--- RUNNING IN TEST MODE ---")
        MIN_DELAY = 0.5
        MAX_DELAY = 1.0

    crawl_mayo_conditions(is_test_mode=args.test)

if __name__ == '__main__':
    main()