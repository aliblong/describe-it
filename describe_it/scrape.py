from datetime import datetime, date, timedelta

import logging
import requests
from bs4 import BeautifulSoup

# Derived from https://github.com/CRutkowski/Kijiji-Scraper
def parse_ad_summary(html):
    '''
    Parses several fields of a listing summary (except the description,
    which is truncated), then returns a dict of several these fields.
    '''
    ad_info = {}

    try:
        ad_info['title'] = html.find('a', {'class': 'title'}).text.strip()
    except:
        logging.error('Unable to parse Title data.')

    try:
        ad_info['img'] = str(html.find('img'))
    except:
        logging.error('Unable to parse Image data')

    try:
        ad_info['url'] = 'http://www.kijiji.ca' + html.get('data-vip-url')
    except:
        logging.error('Unable to parse URL data.')

    try:
        ad_info['details'] = html.find('div', {'class': 'details'}).text.strip()
    except:
        logging.error('Unable to parse Details data.')

    raw_date = ''
    try:
        raw_date = html.find('span', {'class': 'date-posted'}).text.strip()
        # If the ad is less than 24h old, it's displayed as '< some amount of time'
        if raw_date[0] == '<':
            date = datetime.today().date()
        # Or "Yesterday"
        elif raw_date[0] == 'Y':
            date = datetime.today().date() - timedelta(days=1)
        else:
            date = datetime.strptime(raw_date, '%d/%m/%Y')
        ad_info['date_posted'] = date
    except:
        logging.error(f'Unable to parse Date data: \'{raw_date}\'')

    # The location field is affixed with the date for some reason
    try:
        location = html.find('div', {'class': 'location'}).text.strip()
        location = location.replace(raw_date, '')
        ad_info['location'] = location
    except:
        logging.error('Unable to parse Location data.')

    # In addition to normal prices, there can be 'Free' ($0), 'Please Contact' ($-1),
    # some other value ($-2), or empty ($-3)
    try:
        price = html.find('div', {'class': 'price'}).text.strip().split('\n')[0]
        if price[0] == '$':
            price = price[1:]
        elif price == 'Please Contact':
            price = '-1.00'
        elif price == 'Free':
            price = '0'
        else:
            price = '-2.00'
        ad_info['price'] = price
    except:
        logging.error('Unable to parse Price data.')
        ad_info['price'] = '-3.00'

    return ad_info

def parse_ad_page(url):
    '''Parses the description from an ad page'''
    try:
        page = requests.get(url)
    except:
        logging.error('[Error] Unable to load ' + url)
        return None

    soup = BeautifulSoup(page.content, 'html.parser')

    try:
        return '\n'.join([p.get_text() for p in soup.find('div', {'itemprop': 'description'}).find_all('p')])
    except:
        logging.error('Unable to parse ad description.')
        return None

def scrape(subject, existing_ad_ids = None, limit = None, url = None):
    '''
    Args are a search string for some subject, a list of existing ad IDs to skip,
    a limit to the number of results to return, and a url from which to start
    (in case, for example, a previous scrape was halted and is now being resumed).
    Returns (ads, already_scraped_ad_ids, all_ads_scraped); all_ads_scraped is whether
    or not an imposed limit was reached.
    '''
    # Initialize variables for loop
    ad_dict = {}
    # Shallow copy for now
    ad_ids_to_skip = existing_ad_ids if existing_ad_ids is not None else set()

    if url is None: url =         f'https://www.kijiji.ca/b-city-of-toronto/{subject}/k0l1700273?ll=43.650843,-79.377573&dc=true'

    ads_parsed = 0

    already_scraped_ad_ids = set()

    while url:

        try:
            page = requests.get(url)
        except:
            logging.error(f'[Error] Unable to load {url}')
            return

        soup = BeautifulSoup(page.content, 'html.parser')

        ads = soup.find_all('div', {'class': 'regular-ad'})

        # Skip third-party ads; these third parties are typically small retailers
        third_party_ads = soup.find_all('div', {'class': 'third-party'})
        third_party_ad_ids = set([int(ad['data-ad-id']) for ad in third_party_ads])

        # Parse ads until the limit is reached
        for ad in ads:
            title = ad.find('a', {'class': 'title'}).text.strip()
            ad_id = int(ad['data-ad-id'])

            if ad_id not in third_party_ad_ids:
                if ad_id not in ad_ids_to_skip:
                    logging.info(f'New ad found! Ad id: {ad_id}')
                    ad_info = parse_ad_summary(ad)
                    ad_url = ad_info['url']
                    description = parse_ad_page(ad_url)
                    if description is None: continue
                    ad_info['description'] = description
                    ad_dict[ad_id] = ad_info
                    if limit is not None:
                        ads_parsed += 1
                        if ads_parsed >= limit: return ad_dict, already_scraped_ad_ids, url
                else:
                    logging.debug('Skip already-scraped ad')
                    already_scraped_ad_ids.add(ad_id)
            else:
                logging.debug('Skip third-party ad')

        url = soup.find('a', {'title' : 'Next'})
        if url:
            url_path = url['href']
            url = f'https://www.kijiji.ca{url_path}'

    return (ad_dict, already_scraped_ad_ids) if limit is None else (ad_dict, already_scraped_ad_ids, None)
