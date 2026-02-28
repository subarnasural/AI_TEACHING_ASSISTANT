import json
from langchain_chroma import Chroma
from get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma"
K = 5

def text_contains_keywords(text, keywords):
    text = text.lower()
    return any(kw.lower() in text for kw in keywords)

def evaluate():
    with open("eval_dataset.json") as f:
        dataset = json.load(f)

    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embedding_function()
    )

    hits = 0
    reciprocal_ranks = []

    for sample in dataset:
        question = sample["question"]
        relevant_keywords = sample["relevant_keywords"]

        results = db.similarity_search(question, k=K)

        found = False
        rank = 0

        for i, doc in enumerate(results):
            if text_contains_keywords(doc.page_content, relevant_keywords):
                found = True
                rank = i + 1
                break

        hits += int(found)
        reciprocal_ranks.append(1 / rank if rank else 0)

        print(f"\nQ: {question}")
        print(f"Hit@{K}: {found}")
        if found:
            print(f"First relevant rank: {rank}")

    hit_rate = hits / len(dataset)
    mrr = sum(reciprocal_ranks) / len(dataset)

    print("\n===== Retrieval Evaluation =====")
    print(f"Hit@{K}: {hit_rate:.2f}")
    print(f"MRR: {mrr:.2f}")

if __name__ == "__main__":
    evaluate()
