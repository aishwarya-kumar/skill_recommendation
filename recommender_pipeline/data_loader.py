import json
import pandas as pd


def load_market_trends(json_path):
    with open(json_path, 'r') as file:
        return json.load(file)


def load_pay_info(csv_path):
    return pd.read_csv(csv_path)

# Example usage:
# market_trends = load_market_trends("data/market_trends.json")
# pay_info = load_pay_info("data/pay_info.csv")
