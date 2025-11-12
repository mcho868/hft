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
HEALTHIFY_DATA_DIR = '/Users/choemanseung/789/hft/RAGdata/healthify_data'
HEALTHIFY_URL_LIST_FILE = '/Users/choemanseung/789/hft/RAGdata/healthify_urls.json'
HEALTHIFY_EXISTING_URLS_FILE = '/Users/choemanseung/789/hft/RAGdata/healthify_existing_urls.json'

def sanitize_filename(filename):
    """Removes invalid characters from a string to make it a valid filename."""
    return re.sub(r'[\\/*?:"<>|]', "", filename).replace(" ", "_")

def load_existing_urls():
    """Loads the set of already scraped URLs from the healthify_existing_urls.json file."""
    if os.path.exists(HEALTHIFY_EXISTING_URLS_FILE):
        print(f"Loading existing URLs from '{HEALTHIFY_EXISTING_URLS_FILE}'...")
        try:
            with open(HEALTHIFY_EXISTING_URLS_FILE, 'r', encoding='utf-8') as f:
                existing_urls_list = json.load(f)
            existing_urls = set(existing_urls_list)
            print(f"Loaded {len(existing_urls)} existing URLs to skip.")
            return existing_urls
        except Exception as e:
            print(f"Error loading existing URLs: {e}")
            return set()
    else:
        print(f"No existing URLs file found at '{HEALTHIFY_EXISTING_URLS_FILE}'. Will not skip any URLs.")
        return set()

def save_existing_urls(existing_urls):
    """Saves the set of scraped URLs back to the healthify_existing_urls.json file."""
    try:
        existing_urls_list = list(existing_urls)
        with open(HEALTHIFY_EXISTING_URLS_FILE, 'w', encoding='utf-8') as f:
            json.dump(existing_urls_list, f, indent=4, ensure_ascii=False)
        print(f"Updated existing URLs file with {len(existing_urls)} URLs.")
    except Exception as e:
        print(f"Error saving existing URLs: {e}")

def discover_all_healthify_conditions():
    """Discovers all Healthify health condition URLs from the Health A-Z alphabetical pages."""
    base_url = 'https://healthify.nz'
    
    print(f"Discovering all Healthify health conditions from alphabetical pages (a-z)...")
    
    health_urls = set()
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    
    for letter in alphabet:
        health_az_url = f'https://healthify.nz/health-a-z/{letter}'
        print(f"  Checking page: {health_az_url}")
        
        try:
            time.sleep(random.uniform(0.5, 1.0))  # Small delay between requests
            response = requests.get(health_az_url, headers=HEADERS)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find all links to health topics on this alphabetical page
            page_urls = 0
            for link in soup.select('a[href*="/health-a-z/"]'):
                href = link.get('href')
                # Look for URLs like /health-a-z/a/topic-name (not just /health-a-z/a)
                if href and f'/health-a-z/{letter}/' in href and href != f'/health-a-z/{letter}':
                    full_url = urljoin(base_url, href)
                    if full_url not in health_urls:
                        health_urls.add(full_url)
                        page_urls += 1
            
            print(f"    Found {page_urls} new URLs on page {letter}")
            
        except Exception as e:
            print(f"    Error accessing {health_az_url}: {e}")
            continue
    
    health_urls_list = sorted(list(health_urls))
    print(f"Discovered total of {len(health_urls_list)} Healthify health URLs from all alphabetical pages.")
    
    return health_urls_list

def discover_all_healthify_medicines():
    """Discovers all Healthify medicine URLs from the Medicines A-Z alphabetical pages."""
    base_url = 'https://healthify.nz'
    
    print(f"Discovering all Healthify medicines from alphabetical pages (a-z)...")
    
    medicine_urls = set()
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    
    for letter in alphabet:
        medicines_az_url = f'https://healthify.nz/medicines-a-z/{letter}'
        print(f"  Checking page: {medicines_az_url}")
        
        try:
            time.sleep(random.uniform(0.5, 1.0))  # Small delay between requests
            response = requests.get(medicines_az_url, headers=HEADERS)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find all links to medicine topics on this alphabetical page
            page_urls = 0
            for link in soup.select('a[href*="/medicines-a-z/"]'):
                href = link.get('href')
                # Look for URLs like /medicines-a-z/a/medicine-name (not just /medicines-a-z/a)
                if href and f'/medicines-a-z/{letter}/' in href and href != f'/medicines-a-z/{letter}':
                    full_url = urljoin(base_url, href)
                    if full_url not in medicine_urls:
                        medicine_urls.add(full_url)
                        page_urls += 1
            
            print(f"    Found {page_urls} new URLs on page {letter}")
            
        except Exception as e:
            print(f"    Error accessing {medicines_az_url}: {e}")
            continue
    
    medicine_urls_list = sorted(list(medicine_urls))
    print(f"Discovered total of {len(medicine_urls_list)} Healthify medicine URLs from all alphabetical pages.")
    
    return medicine_urls_list

def parse_healthify_content(html, url):
    """Parses Healthify page content and extracts structured text."""
    soup = BeautifulSoup(html, 'html.parser')
    
    # Get the main title
    title_element = soup.select_one('h1')
    title = title_element.get_text(strip=True) if title_element else "Unknown Topic"
    
    # Find the main content area
    main_content = soup.select_one('main') or soup.select_one('.main-content') or soup
    
    # Extract structured content
    sections = []
    
    # Look for collapsible sections first (like the accordions shown in screenshots)
    collapsible_sections = main_content.find_all(['div'], class_=lambda x: x and ('accordion' in ' '.join(x) or 'collaps' in ' '.join(x)))
    
    if collapsible_sections:
        for section in collapsible_sections:
            # Get section header
            header = section.find(['h2', 'h3', 'button', 'a'])
            if header:
                section_title = header.get_text(strip=True)
                sections.append(f"--- {section_title} ---")
                
                # Get section content
                content_elements = section.find_all(['p', 'li', 'div'])
                for element in content_elements:
                    text = element.get_text(strip=True)
                    if text and len(text) > 10:
                        if element.name == 'li':
                            sections.append(f"• {text}")
                        else:
                            sections.append(text)
                sections.append("")
    
    # Also extract any direct content
    content_elements = main_content.find_all(['h2', 'h3', 'p', 'ul', 'li'])
    current_section = None
    section_content = []
    
    for element in content_elements:
        # Skip navigation and non-content elements
        if element.get('class'):
            classes = ' '.join(element.get('class', []))
            if any(skip in classes for skip in ['nav', 'breadcrumb', 'header', 'footer']):
                continue
        
        if element.name in ['h2', 'h3']:
            # Save previous section
            if current_section and section_content:
                sections.append(f"--- {current_section} ---")
                sections.extend(section_content)
                sections.append("")
            
            current_section = element.get_text(strip=True)
            section_content = []
            
        elif element.name == 'p':
            text = element.get_text(strip=True)
            if text and len(text) > 10:
                section_content.append(text)
                
        elif element.name == 'li':
            text = element.get_text(strip=True)
            if text:
                section_content.append(f"• {text}")
    
    # Add the last section
    if current_section and section_content:
        sections.append(f"--- {current_section} ---")
        sections.extend(section_content)
    
    # If no sections found, get basic content
    if not sections:
        basic_content = main_content.find_all(['p', 'li'])
        for element in basic_content:
            text = element.get_text(strip=True)
            if text and len(text) > 10:
                if element.name == 'li':
                    sections.append(f"• {text}")
                else:
                    sections.append(text)
    
    content_text = '\n\n'.join(sections) if sections else None
    return title, content_text

def save_healthify_data(data):
    """Saves a single Healthify page's data to its own JSON file."""
    if not os.path.exists(HEALTHIFY_DATA_DIR):
        os.makedirs(HEALTHIFY_DATA_DIR)
    
    title = data.get("title", "Unknown_Topic")
    filename = sanitize_filename(title) + ".json"
    filepath = os.path.join(HEALTHIFY_DATA_DIR, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"  -> Successfully saved to '{filepath}'")

def crawl_healthify(is_test_mode=False):
    """
    Crawls Healthify health and medicine information with resumability.
    """
    print("\n--- Starting Healthify scraper ---")
    
    # Load existing URLs to skip duplicates
    existing_urls = load_existing_urls()
    
    # Discover or load URLs
    all_urls = []
    
    if os.path.exists(HEALTHIFY_URL_LIST_FILE):
        print(f"Loading URLs from '{HEALTHIFY_URL_LIST_FILE}'...")
        with open(HEALTHIFY_URL_LIST_FILE, 'r', encoding='utf-8') as f:
            all_urls = json.load(f)
        print(f"Loaded {len(all_urls)} URLs.")
    else:
        print("No URL list found. Discovering Healthify URLs...")
        health_urls = discover_all_healthify_conditions()
        medicine_urls = discover_all_healthify_medicines()
        all_urls = health_urls + medicine_urls
        
        # Save the discovered URLs
        with open(HEALTHIFY_URL_LIST_FILE, 'w', encoding='utf-8') as f:
            json.dump(all_urls, f, indent=4, ensure_ascii=False)
        print(f"Discovered total of {len(all_urls)} URLs.")
    
    if not all_urls:
        print("No URLs found. Exiting.")
        return
    
    # Limit URLs for test mode
    urls_to_scrape = all_urls[:5] if is_test_mode else all_urls
    
    print(f"\nStarting to scrape {len(urls_to_scrape)} pages...")
    
    for i, url in enumerate(urls_to_scrape):
        print(f"\n--- Processing page {i+1}/{len(urls_to_scrape)}: {url} ---")
        
        # Skip if already processed
        if url in existing_urls:
            print(f"  -> URL already scraped. Skipping.")
            continue
        
        try:
            # Add delay between requests
            time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))
            
            # Fetch the page
            response = requests.get(url, headers=HEADERS, timeout=30)
            response.raise_for_status()
            
            # Parse the content
            title, content = parse_healthify_content(response.text, url)
            
            if content and content.strip():
                # Determine content type
                content_type = "medicine" if "/medicines-a-z/" in url else "health"
                
                # Save the data
                data = {
                    'source': 'healthify',
                    'type': content_type,
                    'url': url,
                    'title': title,
                    'text': content
                }
                save_healthify_data(data)
                
                # Add to existing URLs and save immediately
                existing_urls.add(url)
                save_existing_urls(existing_urls)
                
                print(f"  -> Successfully processed '{title}' ({content_type})")
            else:
                print(f"  -> No content found for '{title}', skipping.")
                
        except Exception as e:
            print(f"  -> Error processing {url}: {e}")
            continue
    
    print(f"\nCompleted Healthify scraping. Processed {len(existing_urls)} total pages.")

def main():
    parser = argparse.ArgumentParser(description="Scrape Healthify health and medicine information.")
    parser.add_argument('--test', action='store_true', help='Run in test mode (scrape only 5 pages).')
    args = parser.parse_args()

    global MIN_DELAY, MAX_DELAY
    if args.test:
        print("\n--- RUNNING IN TEST MODE ---")
        MIN_DELAY = 0.5
        MAX_DELAY = 1.0

    crawl_healthify(is_test_mode=args.test)

if __name__ == '__main__':
    main()