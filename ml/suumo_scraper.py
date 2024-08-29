import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import arrow
import pandas as pd
import requests
from bs4 import BeautifulSoup
from retry import retry

SUUMO_BASE_URL = "https://suumo.jp/jj/chintai/ichiran/FR301FC001/?fw2=&mt=9999999&cn=9999999&ta=13&et=9999999&sc=13214&shkr1=03&ar=030&bs=040&ct=9999999&shkr3=03&shkr2=03&srch_navi=1&mb=0&shkr4=03&cb=0.0&page={}"
MAX_PAGES_TO_SCRAPE = 52
ML_DIR = Path("ml/")
DATA_DIR = ML_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"

timestamp = arrow.now("Asia/Tokyo").format("YYYYMMDD_HHmmss")
OUTPUT_FILENAME = f"{timestamp}_suumo_rental_properties.csv"
LOG_FILENAME = f"{timestamp}_suumo_scraper.log"

logging.basicConfig(
    filename=DATA_DIR / LOG_FILENAME, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


@dataclass
class PropertyDetails:
    category: str
    name: str
    address: str
    primary_transit_access: str
    age: str
    total_floors: str


@dataclass
class RoomDetails:
    floor: str
    rent: str
    management_fee: str
    deposit: str
    key_money: str
    layout: str
    area: str


@dataclass
class PropertyListing(RoomDetails, PropertyDetails):
    pass


@retry(tries=3, delay=10, backoff=2)
def fetch_suumo_page(url: str) -> str:
    logging.info(f"Fetching SUUMO page: {url}")
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    return response.text


def extract_property_details(property_element: BeautifulSoup) -> PropertyDetails:
    category = property_element.find(class_="ui-pct ui-pct--util1").text.strip()
    name = property_element.find(class_="cassetteitem_content-title").text.strip()
    address = property_element.find(class_="cassetteitem_detail-col1").text.strip()
    transit_access_elements = property_element.find(class_="cassetteitem_detail-col2").find_all(
        class_="cassetteitem_detail-text"
    )
    primary_transit_access = transit_access_elements[0].text.strip()
    age_and_floors = [
        detail.text.strip() for detail in property_element.find(class_="cassetteitem_detail-col3").find_all("div")
    ]
    return PropertyDetails(category, name, address, primary_transit_access, *age_and_floors)


def extract_room_details(room_element: BeautifulSoup) -> RoomDetails:
    room_columns = room_element.find_all("td")
    floor = room_columns[2].text.strip()
    rent = room_columns[3].find(class_="cassetteitem_other-emphasis ui-text--bold").text.strip()
    management_fee = room_columns[3].find(class_="cassetteitem_price cassetteitem_price--administration").text.strip()
    deposit = room_columns[4].find(class_="cassetteitem_price cassetteitem_price--deposit").text.strip()
    key_money = room_columns[4].find(class_="cassetteitem_price cassetteitem_price--gratuity").text.strip()
    layout = room_columns[5].find(class_="cassetteitem_madori").text.strip()
    area = room_columns[5].find(class_="cassetteitem_menseki").text.strip()

    return RoomDetails(floor, rent, management_fee, deposit, key_money, layout, area)


def scrape_suumo_listings(max_pages: int = MAX_PAGES_TO_SCRAPE) -> list[PropertyListing]:
    all_listings = []
    start_time = time.time()

    for page_num in range(1, max_pages + 1):
        print(f"Scraping SUUMO page {page_num}...")

        try:
            page_content = fetch_suumo_page(SUUMO_BASE_URL.format(page_num))
        except requests.exceptions.RequestException:
            logging.exception(f"Failed to fetch SUUMO page {page_num} after retries")
            continue

        soup = BeautifulSoup(page_content, "html.parser")
        property_elements = soup.find_all(class_="cassetteitem")

        if not property_elements:
            logging.info(f"No more properties found on SUUMO page {page_num}. Stopping.")
            break

        for property_element in property_elements:
            try:
                property_details = extract_property_details(property_element)
                room_elements = property_element.find(class_="cassetteitem_other").find_all(class_="js-cassette_link")
                for room_element in room_elements:
                    room_details = extract_room_details(room_element)
                    all_listings.append(PropertyListing(**asdict(property_details), **asdict(room_details)))
            except AttributeError:
                logging.exception(f"Error parsing property data on SUUMO page {page_num}")
                continue

        time.sleep(3)

    elapsed_time = time.time() - start_time
    logging.info(f"SUUMO scraping completed in {elapsed_time:.2f} seconds.")

    return all_listings


def save_listings_to_csv(listings: list[PropertyListing], filename: str = OUTPUT_FILENAME):
    property_df = pd.DataFrame([asdict(listing) for listing in listings])
    property_df.to_csv(RAW_DATA_DIR / filename, index=False)
    logging.info(f"SUUMO listings data saved to {filename}")


if __name__ == "__main__":
    suumo_listings = scrape_suumo_listings()
    save_listings_to_csv(suumo_listings, OUTPUT_FILENAME)
