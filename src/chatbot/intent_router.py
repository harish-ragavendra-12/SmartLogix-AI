"""
SmartLogix AI
Intent Router

Determines which SmartLogix capability should handle
the user's question.
"""


class IntentRouter:

    def classify(self, query):

        query_lower = query.lower().strip()

        # -------------------------------------------------
        # ORDER TRACKING
        # -------------------------------------------------

        order_keywords = [
            "order",
            "delivery",
            "shipment",
            "tracking",
            "track",
            "where is my"
        ]

        if any(keyword in query_lower for keyword in order_keywords):
            return "order_tracking"

        # -------------------------------------------------
        # MAINTENANCE
        # -------------------------------------------------

        maintenance_keywords = [
            "maintenance",
            "vehicle health",
            "vehicle condition",
            "repair",
            "service",
            "failure"
        ]

        if any(keyword in query_lower for keyword in maintenance_keywords):
            return "maintenance"

        # -------------------------------------------------
        # PRODUCT
        # -------------------------------------------------

        product_keywords = [
            "product",
            "price",
            "compare",
            "comparison",
            "recommend",
            "recommendation",
            "buy"
        ]

        if any(keyword in query_lower for keyword in product_keywords):
            return "product"

        # -------------------------------------------------
        # REVIEW
        # -------------------------------------------------

        review_keywords = [
            "review",
            "reviews",
            "customer feedback",
            "sentiment",
            "rating"
        ]

        if any(keyword in query_lower for keyword in review_keywords):
            return "reviews"

        # -------------------------------------------------
        # FAQ / GENERAL
        # -------------------------------------------------

        faq_keywords = [
            "what is",
            "how does",
            "how do",
            "faq",
            "help",
            "smartlogix"
        ]

        if any(keyword in query_lower for keyword in faq_keywords):
            return "faq"

        return "general"