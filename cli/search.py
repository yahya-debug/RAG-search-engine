import string
from nltk.stem import PorterStemmer
from files_data import data, stopWords

stemmer = PorterStemmer()

def search(query: str) -> str :
    result = ""
    i = 1
    for k in data["movies"]:
        parsed_query = query.lower().translate(str.maketrans("", "", string.punctuation)).split(' ')
        title = k["title"].lower().translate(str.maketrans("", "", string.punctuation))
        mathces = False

        for token in parsed_query:
            # stemming the token
            token = stemmer.stem(token)
            if token in stopWords:
                continue
            if token in title:
                mathces = True
        if mathces:
            result += f'{i}. {k["title"]}\n'
            i = i + 1

    return result