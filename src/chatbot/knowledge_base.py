"""
SmartLogix AI
Knowledge Base

Loads selected SmartLogix datasets and provides simple retrieval
functions for the chatbot.
"""

from pathlib import Path
import json
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"


class SmartLogixKnowledgeBase:

    def __init__(self):

        self.orders = self._load_csv(
            DATA_PROCESSED / "orders_cleaned.csv"
        )

        self.customers = self._load_csv(
            DATA_PROCESSED / "customers_cleaned.csv"
        )

        self.maintenance = self._load_csv(
            DATA_PROCESSED / "maintenance_history_clean.csv"
        )

        self.products = self._load_json(
            DATA_RAW / "product_catalog.json"
        )

        self.reviews = self._load_jsonl(
            DATA_RAW / "customer_reviews.jsonl"
        )

    # ---------------------------------------------------------
    # DATA LOADING
    # ---------------------------------------------------------

    @staticmethod
    def _load_csv(path):

        if not path.exists():
            print(f"Warning: Dataset not found: {path}")
            return pd.DataFrame()

        return pd.read_csv(path)

    @staticmethod
    def _load_json(path):

        if not path.exists():
            print(f"Warning: Dataset not found: {path}")
            return []

        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)

    @staticmethod
    def _load_jsonl(path):

        records = []

        if not path.exists():
            print(f"Warning: Dataset not found: {path}")
            return records

        with open(path, "r", encoding="utf-8") as file:

            for line in file:

                line = line.strip()

                if not line:
                    continue

                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        return records

    # ---------------------------------------------------------
    # ORDER SEARCH
    # ---------------------------------------------------------

    def find_order(self, order_id):

        if self.orders.empty:
            return None

        if "order_id" not in self.orders.columns:
            return None

        result = self.orders[
            self.orders["order_id"].astype(str).str.upper()
            == str(order_id).upper()
        ]

        if result.empty:
            return None

        return result.iloc[0].to_dict()

    # ---------------------------------------------------------
    # PRODUCT SEARCH
    # ---------------------------------------------------------

    def search_products(self, query):

        if not self.products:
            return []

        query = str(query).lower()

        matches = []

        for product in self.products:

            text = json.dumps(
                product,
                ensure_ascii=False
            ).lower()

            if query in text:
                matches.append(product)

        return matches[:5]

    # ---------------------------------------------------------
    # MAINTENANCE SEARCH
    # ---------------------------------------------------------

    def find_vehicle_maintenance(self, vehicle_id):

        if self.maintenance.empty:
            return []

        if "vehicle_id" not in self.maintenance.columns:
            return []

        result = self.maintenance[
            self.maintenance["vehicle_id"].astype(str).str.upper()
            == str(vehicle_id).upper()
        ]

        return result.to_dict(orient="records")

    # ---------------------------------------------------------
    # REVIEW SEARCH
    # ---------------------------------------------------------

    def search_reviews(self, query):

        if not self.reviews:
            return []

        query = str(query).lower()

        matches = []

        for review in self.reviews:

            text = json.dumps(
                review,
                ensure_ascii=False
            ).lower()

            if query in text:
                matches.append(review)

        return matches[:5]

    # ---------------------------------------------------------
    # DATASET SUMMARY
    # ---------------------------------------------------------

    def summary(self):

        return {
            "orders": len(self.orders),
            "customers": len(self.customers),
            "maintenance_records": len(self.maintenance),
            "products": len(self.products),
            "reviews": len(self.reviews),
        }