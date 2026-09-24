"""
SmartLogix AI
Chatbot MVP Test
"""

from chatbot.smartlogix_assistant import SmartLogixAssistant


def main():

    print("=" * 70)
    print("SMARTLOGIX AI - LLM / RAG ASSISTANT MVP")
    print("=" * 70)

    assistant = SmartLogixAssistant()

    if not assistant.knowledge_base.orders.empty:
        real_order_id = (
            assistant.knowledge_base.orders.iloc[0]["order_id"]
        )

        print("\nReal order ID available for testing:")
        print(real_order_id)

    print("\nKnowledge Base:")
    print(
        assistant.knowledge_base.summary()
    )

    test_queries = [
        f"Track order {real_order_id}",
        "Show maintenance information for VEH-0650",
        "What products are available?",
        "Tell me about SmartLogix",
        "I need help with an order"
    ]

    for query in test_queries:

        print("\n" + "=" * 70)
        print("USER")
        print("=" * 70)

        print(query)

        result = assistant.ask(query)

        print("\nINTENT:")
        print(result["intent"])

        print("\nASSISTANT:")
        print(result["response"])

    print("\n" + "=" * 70)
    print("CHATBOT MVP TEST COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()