import logging
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup
from retry import retry

BASE_URL = "https://suumo.jp/jj/chintai/ichiran/FR301FC001/?fw2=&mt=9999999&cn=9999999&ta=13&et=9999999&sc=13214&shkr1=03&ar=030&bs=040&ct=9999999&shkr3=03&shkr2=03&srch_navi=1&mb=0&shkr4=03&cb=0.0&page={}"
MAX_PAGES = 52
OUTPUT_CSV = "property_data.csv"

logging.basicConfig(
    filename="ml/data/scrape.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


@retry(tries=3, delay=10, backoff=2)
def fetch_page_content(url):
    logging.info(f"Fetching URL: {url}")
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    return response.text


def parse_property_info(property_info):
    category = property_info.find(class_="ui-pct ui-pct--util1").text.strip()
    property_name = property_info.find(class_="cassetteitem_content-title").text.strip()
    address = property_info.find(class_="cassetteitem_detail-col1").text.strip()
    # transit_access_details = [
    #     transit.text.strip()
    #     for transit in property_info.find(class_="cassetteitem_detail-col2").find_all(class_="cassetteitem_detail-text")
    # ]
    transit_access_details = property_info.find(class_="cassetteitem_detail-col2").find_all(
        class_="cassetteitem_detail-text"
    )
    primary_transit_access = transit_access_details[0].text.strip() if transit_access_details else None
    age_and_floors = [
        detail.text.strip() for detail in property_info.find(class_="cassetteitem_detail-col3").find_all("div")
    ]
    # return [category, property_name, address, *transit_access_details, *age_and_floors]
    return [category, property_name, address, primary_transit_access, *age_and_floors]


def parse_room_info(room_info):
    room_columns = room_info.find_all("td")
    room_floor = room_columns[2].text.strip()
    room_rent = room_columns[3].find(class_="cassetteitem_other-emphasis ui-text--bold").text.strip()
    room_management_fee = (
        room_columns[3].find(class_="cassetteitem_price cassetteitem_price--administration").text.strip()
    )
    room_deposit = room_columns[4].find(class_="cassetteitem_price cassetteitem_price--deposit").text.strip()
    room_gratuity = room_columns[4].find(class_="cassetteitem_price cassetteitem_price--gratuity").text.strip()
    room_layout = room_columns[5].find(class_="cassetteitem_madori").text.strip()
    room_area = room_columns[5].find(class_="cassetteitem_menseki").text.strip()

    return [room_floor, room_rent, room_management_fee, room_deposit, room_gratuity, room_layout, room_area]


def scrape_suumo_properties(max_pages=MAX_PAGES):
    all_property_room_data = []
    start_time = time.time()

    for page_num in range(1, max_pages + 1):
        print(f"Scraping page {page_num}...")

        try:
            page_content = fetch_page_content(BASE_URL.format(page_num))
        except requests.exceptions.RequestException:
            logging.exception(f"Failed to fetch page {page_num} after retries")
            continue

        soup = BeautifulSoup(page_content, "html.parser")
        property_list = soup.find_all(class_="cassetteitem")

        if not property_list:
            logging.info(f"No more properties found at page {page_num}. Stopping.")
            break

        for property_info in property_list:
            try:
                property_details = parse_property_info(property_info)
                room_list = property_info.find(class_="cassetteitem_other")
                for room_info in room_list.find_all(class_="js-cassette_link"):
                    room_details = parse_room_info(room_info)
                    all_property_room_data.append(property_details + room_details)
            except AttributeError:
                logging.exception(f"Error parsing property data on page {page_num}")
                continue

        time.sleep(3)

    end_time = time.time()
    elapsed_time = end_time - start_time
    logging.info(f"Scraping completed in {elapsed_time:.2f} seconds.")

    return all_property_room_data


def save_to_csv(data, filename=OUTPUT_CSV):
    property_df = pd.DataFrame(
        data,
        columns=[
            "Category",
            "Property Name",
            "Address",
            "Primary Transit Access",
            "Age",
            "Floors",
            "Room Floor",
            "Rent",
            "Management Fee",
            "Deposit",
            "Gratuity",
            "Layout",
            "Area",
        ],
    )
    property_df.to_csv(f"ml/data/raw/{filename}", index=False)
    logging.info(f"Data saved to {filename}")


if __name__ == "__main__":
    properties_data = scrape_suumo_properties()
    save_to_csv(properties_data)
