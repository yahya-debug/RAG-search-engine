import argparse
import json
import os
from lib.hybrid_search import HybridSearch, rrf_search
from files_data import load_movies

GOLDEN_DS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "golden_dataset.json")

global golden_dataset
with open(GOLDEN_DS_PATH, "r") as file:
    golden_dataset = json.load(file)

documents = load_movies()['movies']
def main() -> None:
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument("--limit",type=int,default=5,help="Number of results to evaluate (k for precision@k, recall@k)",)

    args = parser.parse_args()
    limit = args.limit

    # run evaluation logic here
    print(f"k={limit}\n\n")
    for case in golden_dataset['test_cases']:
        search_res = rrf_search(HybridSearch(documents), case['query'], 60, limit)
        found = 0
        for res in search_res:
            if documents[res[0]-1]['title'] in case['relevant_docs']:
                found+=1
        
        percision = found/len(search_res)
        recall = found/len(case['relevant_docs'])
        print(f"- Query: {case['query']}\n\t- Precision@{limit}: {percision:.4f}\Recall@{limit}: {recall:.4f}\nF1 Score: {2*(percision * recall)/(percision + recall):.4f}\n- Retrieved: {', '.join([documents[res[0]-1]['title'] for res in search_res])}\n- Relevant: {' '.join(case['relevant_docs'])}")

if __name__ == "__main__":
    main()
