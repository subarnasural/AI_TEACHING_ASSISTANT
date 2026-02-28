from query_data import query_rag
from langchain_ollama import ChatOllama


EVAL_PROMPT = """
Expected Response: {expected_response}
Actual Response: {actual_response}
---
(Answer with 'true' or 'false') Does the actual response match the expected response?
"""


def test_monopoly_rules():
    assert query_and_validate(
        question="How much total money does a player start with in Monopoly? (Answer with the number only)",
        expected_response="$1500",
    )


def test_ticket_to_ride_rules():
    assert query_and_validate(
        question="How many points does the longest continuous train get in Ticket to Ride? (Answer with the number only)",
        expected_response="10",
    )


def query_and_validate(question: str, expected_response: str):
    # Run RAG
    response_text = query_rag(question)

    # Build evaluation prompt
    prompt = EVAL_PROMPT.format(
        expected_response=expected_response,
        actual_response=response_text,
    )

    # LLM judge
    llm = ChatOllama(
        model="mistral",
        base_url="http://localhost:11434",
    )

    evaluation_message = llm.invoke(prompt)
    evaluation_results = evaluation_message.content.strip().lower()

    print("\nEvaluation Prompt:\n", prompt)
    print("LLM Judge Output:", evaluation_results)

    if "true" in evaluation_results:
        print("\033[92mResult: TRUE\033[0m")
        return True
    elif "false" in evaluation_results:
        print("\033[91mResult: FALSE\033[0m")
        return False
    else:
        raise ValueError("Invalid evaluation result from LLM.")
