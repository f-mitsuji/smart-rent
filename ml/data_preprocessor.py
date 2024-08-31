import re
from pathlib import Path

import numpy as np
import pandas as pd
from ml_settings import PROCESSED_DATA_DIR, RAW_DATA_DIR

RAW_DATA_PATH = RAW_DATA_DIR / "suumo_rental_properties.csv"
PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "suumo_rental_properties_cleaned2.csv"


def clean_numeric_columns(property_df: pd.DataFrame) -> pd.DataFrame:
    property_df["rent"] = property_df["rent"].str.replace("万円", "").astype(float)
    property_df["management_fee"] = property_df["management_fee"].str.replace("円", "").replace("-", "0").astype(float)
    property_df["deposit"] = property_df["deposit"].str.replace("万円", "").replace("-", "0").astype(float)
    property_df["key_money"] = property_df["key_money"].str.replace("万円", "").replace("-", "0").astype(float)
    property_df["area"] = property_df["area"].str.replace("m2", "").astype(float)
    property_df["building_age"] = property_df["age"].str.replace("新築", "1").str.extract(r"(\d+)").astype(int)
    property_df["total_floors"] = (
        property_df["total_floors"].str.replace("平屋", "1階建").str.extract(r"(\d+)階建").astype(int)
    )
    return property_df


def process_floor_data(property_df: pd.DataFrame) -> pd.DataFrame:
    def parse_floor(floor_value: str) -> float:
        if pd.isna(floor_value) or floor_value == "-":
            return np.nan
        try:
            if "地下" in floor_value:
                return -float(floor_value.replace("地下", "").replace("階", ""))
            if "-" in floor_value:
                floor_min, floor_max = map(float, floor_value.replace("階", "").split("-"))
                return (floor_min + floor_max) / 2
            return float(floor_value.replace("階", ""))
        except ValueError:
            print(f"Warning: Unable to parse floor value: {floor_value}")
            return np.nan

    property_df["floor"] = property_df["floor"].apply(parse_floor)
    median_floor = property_df["floor"].median()
    property_df["floor"] = property_df["floor"].fillna(median_floor)
    return property_df


def extract_address_components(property_df: pd.DataFrame) -> pd.DataFrame:
    property_df["prefecture"] = property_df["address"].str.extract(r"^(.+?[都道府県])")
    property_df["city"] = property_df["address"].str.extract(r"[都道府県]\s*([^市区町村]+?[市区町村])")
    property_df["town"] = property_df["address"].str.extract(r"[市区町村]\s*(.+)")
    property_df = pd.get_dummies(property_df, columns=["town"], drop_first=True)
    return property_df.drop(columns=["prefecture", "city", "address"])  # 国分寺市のデータのみを抽出しているため


def parse_transit_info(transit_value: str) -> tuple[str, str, int, int]:
    line_station_match = re.search(r"(.+)/(.+?)駅", transit_value)
    line = line_station_match.group(1) if line_station_match else None
    station = line_station_match.group(2) if line_station_match else None

    bus_time_match = re.search(r"バス(\d+)分", transit_value)
    bus_used = 1 if bus_time_match else 0
    bus_minutes = int(bus_time_match.group(1)) if bus_time_match else 0

    walk_time_match = re.search(r"歩(\d+)分", transit_value)
    walk_minutes = int(walk_time_match.group(1)) if walk_time_match else 0

    total_time = bus_minutes + walk_minutes
    return line, station, bus_used, total_time


def process_transit_access(property_df: pd.DataFrame) -> pd.DataFrame:
    transit_info = property_df["primary_transit_access"].apply(parse_transit_info)
    property_df[["line", "station", "bus_used", "total_transit_time"]] = pd.DataFrame(
        transit_info.tolist(), index=property_df.index
    )
    property_df = pd.get_dummies(property_df, columns=["line", "station"], drop_first=True)
    return property_df.drop(columns=["primary_transit_access"])


def preprocess_suumo_data(input_path: Path, output_path: Path) -> None:
    """Main function to preprocess SUUMO rental data."""
    property_df = pd.read_csv(input_path)

    property_df = clean_numeric_columns(property_df)
    property_df = process_floor_data(property_df)
    property_df = extract_address_components(property_df)
    property_df = process_transit_access(property_df)

    property_df = pd.get_dummies(property_df, columns=["category", "layout"], drop_first=True)
    property_df = property_df.drop(columns=["name", "age"])

    check_remaining_nans(property_df)

    print(f"Preprocessed data shape: {property_df.shape}")
    property_df.to_csv(output_path, index=False)
    print(f"Preprocessed data saved to {output_path}")


def check_remaining_nans(property_df: pd.DataFrame) -> None:
    nan_columns = property_df.columns[property_df.isna().any()].tolist()
    if nan_columns:
        print("Warning: The following columns still contain NaN values:")
        for col in nan_columns:
            nan_count = property_df[col].isna().sum()
            nan_percentage = (nan_count / len(property_df)) * 100
            print(f"  {col}: {nan_count} NaNs ({nan_percentage:.2f}%)")


if __name__ == "__main__":
    preprocess_suumo_data(RAW_DATA_PATH, PROCESSED_DATA_PATH)
