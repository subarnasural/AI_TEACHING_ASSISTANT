import json
from query_data import query_rag

def keyword_score(answer_text, keywords):
    answer_text = answer_text.lower()
    hits = sum(1 for kw in keywords if kw.lower() in answer_text)
    return hits / len(keywords)

def evaluate():
    with open("eval_dataset.json") as f:
        dataset = json.load(f)

    scores = []

    for sample in dataset:
        question = sample["question"]
        keywords = sample["expected_keywords"]

        answer = query_rag(question)      # AIMessage
        answer_text = answer.content      # ✅ extract text

        score = keyword_score(answer_text, keywords)
        scores.append(score)

        print("\n==============================")
        print(f"Q: {question}")
        print(f"Answer relevance score: {score:.2f}")

    avg_score = sum(scores) / len(scores)
    print("\n===== Generation Evaluation =====")
    print(f"Average Answer Relevance: {avg_score:.2f}")

if __name__ == "__main__":
    evaluate()
