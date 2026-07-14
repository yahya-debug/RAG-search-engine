from PIL import Image
from sentence_transformers import SentenceTransformer
from lib.semantic_search import cosine_similarity
from files_data import load_movies

class MultimodalSearch:
    def __init__(self, documents, model_name="clip-ViT-B-32"):
        self.model = SentenceTransformer(model_name)
        self.texts = []
        self.documents = documents

        for doc in documents:
            self.texts.append(f"{doc["title"]}: {doc['description']}")

        self.text_embeddings = self.model.encode(self.texts)


    def search_with_image(self, img_path):
        # encode image
        img = Image.open(img_path)
        img_embed = self.model.encode([ img ])

        # cosine similarity between the image and the texts embeddings
        scoring = self.documents
        i = 0
        for embed in scoring:
            embed['score'] = cosine_similarity(self.text_embeddings[i], img_embed[0])
            i+=1

        sorted_res = sorted(scoring, key=lambda x: x['score'], reverse=True)

        return sorted_res[:5]

    def embed_image(self, img_path):
        img = Image.open(img_path)
        encoded = self.model.encode([ img ])
        return encoded[0]
    

def verify_image_embedding(img_path):
    multi_modal = MultimodalSearch()

    embedding = multi_modal.embed_image(img_path=img_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")


def image_search_command(img_path):
    documents = load_movies()['movies']
    multi_modal = MultimodalSearch(documents=documents)

    return multi_modal.search_with_image(img_path=img_path)