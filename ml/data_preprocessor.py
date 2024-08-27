import re

import numpy as np
import pandas as pd


def load_property_data(filepath):
    property_df = pd.read_csv(filepath)

    property_df["Rent"] = property_df["Rent"].str.replace("万円", "").astype(float)
    property_df["Management Fee"] = (
        property_df["Management Fee"].str.replace("円", "").str.replace("-", "0").astype(float)
    )
    property_df["Deposit"] = property_df["Deposit"].str.replace("万円", "").str.replace("-", "0").astype(float)
    property_df["Gratuity"] = property_df["Gratuity"].str.replace("万円", "").str.replace("-", "0").astype(float)
    property_df["Area"] = property_df["Area"].str.replace("m2", "").astype(float)

    property_df["Building Age"] = property_df["Age"].str.replace("新築", "1").str.extract(r"(\d+)").astype(int)
    property_df["Total Floors"] = (
        property_df["Floors"].str.replace("平屋", "1階建").str.extract(r"(\d+)階建").astype(int)
    )

    property_df = pd.get_dummies(property_df, columns=["Category", "Layout"], drop_first=True)

    property_df = property_df.drop(columns=["Property Name", "Age", "Floors"])

    return property_df  # noqa: RET504


def parse_room_floor(value):
    try:
        if "地下" in value:
            return -int(value.replace("地下", "").replace("階", ""))
        if "-" in value:
            floor_min, floor_max = map(int, value.replace("階", "").split("-"))
            return (floor_min + floor_max) / 2
        if value == "-":
            return np.nan
        return int(value.replace("階", ""))
    except ValueError:
        print(f"Error converting floor value: {value}")
        return np.nan


def extract_address_components(property_df):
    property_df["Prefecture"] = property_df["Address"].str.extract(r"^(.+?[都道府県])")
    property_df["City"] = property_df["Address"].str.extract(r"[都道府県]\s*([^市区町村]+?[市区町村])")
    property_df["Town"] = property_df["Address"].str.extract(r"[市区町村]\s*(.+)")

    property_df = pd.get_dummies(property_df, columns=["Town"], drop_first=True)
    property_df = property_df.drop(columns=["Prefecture", "City", "Address"])

    return property_df  # noqa: RET504


def parse_transit_info(value):
    match = re.search(r"(.+)/(.+?)駅", value)
    line = match.group(1) if match else None
    station = match.group(2) if match else None

    bus_time = re.search(r"バス(\d+)分", value)
    bus_used = 1 if bus_time else 0
    bus_minutes = int(bus_time.group(1)) if bus_time else 0

    walk_time = re.search(r"歩(\d+)分", value)
    walk_minutes = int(walk_time.group(1)) if walk_time else 0

    total_time = bus_minutes + walk_minutes

    return line, station, bus_used, total_time


def process_transit_access(property_df):
    property_df[["Line", "Station", "Bus Used", "Total Transit Time"]] = property_df["Primary Transit Access"].apply(
        lambda x: pd.Series(parse_transit_info(x))
    )

    property_df = pd.get_dummies(property_df, columns=["Line", "Station"], drop_first=True)
    property_df = property_df.drop(columns=["Primary Transit Access"])

    return property_df  # noqa: RET504


def main():
    filepath = "ml/data/raw/property_data.csv"
    property_df = load_property_data(filepath)

    property_df["Room Floor"] = property_df["Room Floor"].apply(parse_room_floor)

    property_df = extract_address_components(property_df)

    property_df = process_transit_access(property_df)

    print(property_df.head())

    processed_filepath = "ml/data/processed/property_data_cleaned.csv"
    property_df.to_csv(processed_filepath, index=False)


if __name__ == "__main__":
    main()
