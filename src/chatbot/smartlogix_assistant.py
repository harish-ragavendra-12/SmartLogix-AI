"""
SmartLogix AI
Multi-Intent Logistics Assistant

Combines:
    - Intent routing
    - Dataset retrieval
    - Response generation

Current implementation provides the retrieval/tool-routing
foundation for the SmartLogix AI assistant.
"""

import re

from chatbot.knowledge_base import SmartLogixKnowledgeBase
from chatbot.intent_router import IntentRouter


class SmartLogixAssistant:

    def __init__(self):

        self.knowledge_base = SmartLogixKnowledgeBase()
        self.router = IntentRouter()

    # =========================================================
    # MAIN QUERY FUNCTION
    # =========================================================

    def ask(self, query):

        intent = self.router.classify(query)

        if intent == "order_tracking":
            response = self._handle_order(query)

        elif intent == "maintenance":
            response = self._handle_maintenance(query)

        elif intent == "product":
            response = self._handle_product(query)

        elif intent == "reviews":
            response = self._handle_reviews(query)

        elif intent == "faq":
            response = self._handle_faq(query)

        else:
            response = self._handle_general(query)

        return {
            "query": query,
            "intent": intent,
            "response": response
        }

    # =========================================================
    # ORDER TRACKING
    # =========================================================

    def _handle_order(self, query):

        order_id = self._extract_order_id(query)

        if not order_id:

            return (
                "I can help track a SmartLogix order. "
                "Please provide the order ID."
            )

        order = self.knowledge_base.find_order(order_id)

        if not order:

            return (
                f"I could not find order {order_id}. "
                "Please verify the order ID."
            )

        response = f"Order {order_id} information:\n"

        important_fields = [
            "order_status",
            "transport_mode",
            "origin",
            "destination",
            "package_weight_kg",
            "payment_mode"
        ]

        for field in important_fields:

            if field in order:

                value = order[field]

                response += (
                    f"- {field.replace('_', ' ').title()}: "
                    f"{value}\n"
                )

        return response.strip()

    # =========================================================
    # MAINTENANCE
    # =========================================================

    def _handle_maintenance(self, query):

        vehicle_id = self._extract_vehicle_id(query)

        if not vehicle_id:

            return (
                "I can provide vehicle maintenance information. "
                "Please provide a vehicle ID such as VEH-0650."
            )

        records = self.knowledge_base.find_vehicle_maintenance(
            vehicle_id
        )

        if not records:

            return (
                f"No maintenance records were found for "
                f"{vehicle_id}."
            )

        response = (
            f"Maintenance information for {vehicle_id}:\n"
        )

        response += f"- Records found: {len(records)}\n"

        # -----------------------------------------------------
        # FAILURE INFORMATION
        # -----------------------------------------------------

        if "failure_reported" in records[0]:

            failures = sum(
                1
                for record in records
                if str(
                    record.get("failure_reported")
                ).lower()
                == "yes"
            )

            response += (
                f"- Reported failures: {failures}\n"
            )

        # -----------------------------------------------------
        # MAINTENANCE COST
        # -----------------------------------------------------

        if "cost_inr" in records[0]:

            total_cost = sum(
                float(record.get("cost_inr", 0))
                for record in records
            )

            response += (
                f"- Total maintenance cost: "
                f"₹{total_cost:,.2f}\n"
            )

        return response.strip()

    # =========================================================
    # PRODUCT
    # =========================================================

    def _handle_product(self, query):

        search_terms = self._extract_search_terms(
            query
        ).strip()

        # -----------------------------------------------------
        # GENERIC PRODUCT CATALOG REQUEST
        # -----------------------------------------------------

        generic_terms = [
            "",
            "available",
            "available?",
            "list",
            "all",
            "all?"
        ]

        if search_terms.lower() in generic_terms:

            products = self.knowledge_base.products

            # -------------------------------------------------
            # PRODUCT CATALOG IS A DICTIONARY
            # -------------------------------------------------

            if isinstance(products, dict):

                # Actual product records are stored under
                # the "products" key.
                if isinstance(
                    products.get("products"),
                    list
                ):

                    product_items = (
                        products["products"][:5]
                    )

                    if not product_items:

                        return (
                            "No products are currently "
                            "available."
                        )

                    response = (
                        "SmartLogix product catalog:\n"
                    )

                    for index, product in enumerate(
                        product_items,
                        start=1
                    ):

                        response += (
                            f"\nProduct {index}:\n"
                        )

                        if isinstance(product, dict):

                            fields = [
                                "product_id",
                                "product_name",
                                "category",
                                "sub_category",
                                "price",
                                "stock_qty",
                                "avg_rating"
                            ]

                            for field in fields:

                                if field in product:

                                    value = product[field]

                                    response += (
                                        f"- "
                                        f"{field.replace('_', ' ').title()}: "
                                        f"{value}\n"
                                    )

                        else:

                            response += (
                                f"- Details: {product}\n"
                            )

                    return response.strip()

                # -------------------------------------------------
                # FALLBACK FOR UNEXPECTED DICTIONARY STRUCTURE
                # -------------------------------------------------

                product_items = list(
                    products.items()
                )[:5]

                if not product_items:

                    return (
                        "No products are currently "
                        "available."
                    )

                response = (
                    "SmartLogix product catalog:\n"
                )

                for index, (
                    product_id,
                    product_data
                ) in enumerate(
                    product_items,
                    start=1
                ):

                    response += (
                        f"\nProduct {index}:\n"
                    )

                    response += (
                        f"- Product ID: "
                        f"{product_id}\n"
                    )

                    response += (
                        f"- Details: "
                        f"{product_data}\n"
                    )

                return response.strip()

            # -------------------------------------------------
            # PRODUCT CATALOG IS A LIST
            # -------------------------------------------------

            elif isinstance(products, list):

                product_items = products[:5]

                if not product_items:

                    return (
                        "No products are currently "
                        "available."
                    )

                response = (
                    "SmartLogix product catalog:\n"
                )

                for index, product in enumerate(
                    product_items,
                    start=1
                ):

                    response += (
                        f"\nProduct {index}:\n"
                    )

                    if isinstance(product, dict):

                        fields = [
                            "product_id",
                            "product_name",
                            "category",
                            "sub_category",
                            "price",
                            "stock_qty",
                            "avg_rating"
                        ]

                        for field in fields:

                            if field in product:

                                value = product[field]

                                response += (
                                    f"- "
                                    f"{field.replace('_', ' ').title()}: "
                                    f"{value}\n"
                                )

                    else:

                        response += (
                            f"- Details: {product}\n"
                        )

                return response.strip()

            return (
                "Unable to read the product "
                "catalog format."
            )

        # =====================================================
        # SPECIFIC PRODUCT SEARCH
        # =====================================================

        matches = self.knowledge_base.search_products(
            search_terms
        )

        if not matches:

            return (
                f"No products were found for "
                f"'{search_terms}'."
            )

        response = (
            f"Found {len(matches)} matching "
            f"product(s):\n"
        )

        for index, product in enumerate(
            matches,
            start=1
        ):

            response += (
                f"\nProduct {index}:\n"
            )

            if isinstance(product, dict):

                fields = [
                    "product_id",
                    "product_name",
                    "category",
                    "sub_category",
                    "price",
                    "stock_qty",
                    "avg_rating"
                ]

                for field in fields:

                    if field in product:

                        value = product[field]

                        response += (
                            f"- "
                            f"{field.replace('_', ' ').title()}: "
                            f"{value}\n"
                        )

            else:

                response += (
                    f"- {product}\n"
                )

        return response.strip()

    # =========================================================
    # REVIEWS
    # =========================================================

    def _handle_reviews(self, query):

        search_terms = self._extract_search_terms(
            query
        ).strip()

        if not search_terms:

            return (
                "Please provide a product name or "
                "keyword for review analysis."
            )

        matches = self.knowledge_base.search_reviews(
            search_terms
        )

        if not matches:

            return (
                f"No customer reviews were found "
                f"for '{search_terms}'."
            )

        return (
            f"I found {len(matches)} relevant "
            f"customer review(s) for "
            f"'{search_terms}'. "
            "These reviews can be passed to the "
            "sentiment and summarization layer "
            "for deeper analysis."
        )

    # =========================================================
    # FAQ
    # =========================================================

    def _handle_faq(self, query):

        query_lower = query.lower()

        if "smartlogix" in query_lower:

            return (
                "SmartLogix AI is an intelligent "
                "logistics platform that combines "
                "machine learning, route optimization, "
                "computer vision, predictive maintenance, "
                "and AI-assisted logistics operations."
            )

        if "how does" in query_lower:

            return (
                "SmartLogix processes logistics data, "
                "predicts transportation and delivery "
                "outcomes, optimizes routes and vehicle "
                "assignments, monitors vehicle maintenance, "
                "and provides AI-assisted responses."
            )

        return (
            "SmartLogix AI provides logistics intelligence "
            "including delivery prediction, ETA estimation, "
            "route optimization, vehicle assignment, "
            "predictive maintenance, computer vision, "
            "and AI-assisted customer support."
        )

    # =========================================================
    # GENERAL
    # =========================================================

    def _handle_general(self, query):

        return (
            "I am the SmartLogix AI logistics assistant. "
            "I can help with order tracking, vehicle "
            "maintenance, products, customer reviews, "
            "and logistics information."
        )

    # =========================================================
    # ORDER ID EXTRACTION
    # =========================================================

    @staticmethod
    def _extract_order_id(query):

        # Matches IDs such as:
        # ORD-011545
        # ORD-SAMPLE-001
        # ORD-ABC-123

        pattern = (
            r"\bORD-[A-Z0-9]+(?:-[A-Z0-9]+)*\b"
        )

        match = re.search(
            pattern,
            query.upper()
        )

        if match:
            return match.group(0)

        return None

    # =========================================================
    # VEHICLE ID EXTRACTION
    # =========================================================

    @staticmethod
    def _extract_vehicle_id(query):

        # Matches IDs such as:
        # VEH-0650
        # VEH-0274

        pattern = (
            r"\bVEH-[A-Z0-9]+(?:-[A-Z0-9]+)*\b"
        )

        match = re.search(
            pattern,
            query.upper()
        )

        if match:
            return match.group(0)

        return None

    # =========================================================
    # SEARCH TERM EXTRACTION
    # =========================================================

    @staticmethod
    def _extract_search_terms(query):

        query = query.lower()

        remove_words = [
            "what",
            "is",
            "are",
            "the",
            "product",
            "products",
            "price",
            "compare",
            "comparison",
            "recommend",
            "recommendation",
            "review",
            "reviews",
            "customer",
            "feedback",
            "sentiment",
            "rating",
            "please",
            "show",
            "me",
            "about",
            "for",
            "of",
            "in"
        ]

        words = query.split()

        filtered = [
            word
            for word in words
            if word not in remove_words
        ]

        return " ".join(filtered).strip()