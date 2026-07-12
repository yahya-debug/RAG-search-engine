import string
from files_data import load_movies, tokenize_text
from lib.keyword_search import InvertedIndex

data = load_movies()



class Search:

    def __init__(self, index: InvertedIndex):
        self.index = index

    def search(self, query: str) -> str :
        result = []
        i = 1
        parsed_query = tokenize_text(query)
        
        match_set = set()
        for token in parsed_query:
            match_set.update(self.index.get_documents(token))

        match_set = sorted(match_set)
        
        i = 0
        for id in match_set:
            if i == 5:
                break

            movie = self.index.docmap.get(id)
            result.append(f"{movie["id"]} - {movie['title']}")
            i += 1

        return "\n".join(result)
