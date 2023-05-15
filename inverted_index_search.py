import pymorphy2
import re
import pandas as pd

from functools import lru_cache


class Tokenizer:
    def tokenize(self, text: str) -> list:
        return text.split(" ")


class PymorphyTokenizer(Tokenizer):
    def __init__(self):
        self.morph = pymorphy2.MorphAnalyzer()

    @lru_cache(1000000)
    def tokenize(self, text: str) -> list:
        return [
            self.normal_forms(token) for token in re.split("\W+", text.lower()) if token
        ]

    @lru_cache(1000000)
    def normal_forms(self, token: str):
        return self.morph.normal_forms(token)[0]


from collections import defaultdict

# без инвертированного индекса
class SearchEngine:
    def __init__(self, tokenizer: Tokenizer):
        self.tok = tokenizer
        self.docs = {}
        self.doc2text = {}

    def add_document(self, doc_id: int, text: str):
        tokens = set(self.tok.tokenize(text))
        self.docs[doc_id] = tokens
        self.doc2text[doc_id] = text

    def search(self, query: str):
        query_tokens = set(self.tok.tokenize(query))
        doc_ids = []
        for doc_id, doc_tokens in self.docs.items():
            if query_tokens <= doc_tokens:
                doc_ids.append(doc_id)

        return [self.doc2text[doc_id] for doc_id in doc_ids]


class SmartSearchEngine:
    def __init__(self, tokenizer: Tokenizer):
        self.tok = tokenizer
        self.inverted_index = defaultdict(set)
        self.doc2text = {}

    def add_document(self, doc_id: int, text: str):
        tokens = self.tok.tokenize(text)
        for token in tokens:
            self.inverted_index[token].add(doc_id)

        self.doc2text[doc_id] = text

    def search(self, query: str):
        query_tokens = self.tok.tokenize(query)
        doc_ids = None
        for token in query_tokens:
            token_doc_ids = self.inverted_index[token]  # {0, 1, 2}
            if doc_ids is None:
                doc_ids = token_doc_ids
            else:
                doc_ids &= token_doc_ids

        return [self.doc2text[doc_id] for doc_id in doc_ids]


# now: {token: {0, 1, 2}}
# we want: {token: {0: 1, 1: 3, 2: 4}}


class MoreSmartSearchEngine:
    def __init__(self, tokenizer: Tokenizer):
        self.tok = tokenizer
        self.inverted_index = defaultdict(dict)
        self.doc2text = {}

    def add_document(self, doc_id: int, text: str):
        tokens = self.tok.tokenize(text)
        for position, token in reversed(list(enumerate(tokens))):
            self.inverted_index[token][doc_id] = position

        self.doc2text[doc_id] = text

    def search(self, query: str):
        query_tokens = self.tok.tokenize(query)
        doc_ids = None
        doc_matches_count = None
        for token in query_tokens:
            token_doc_ids = self.inverted_index[token]  # {0: 2, 1: 3, 2: 2}
            if doc_ids is None:
                doc_ids = token_doc_ids
                doc_matches_count = {doc_id: 1 for doc_id in doc_ids}
            else:
                for doc_id, position in token_doc_ids.items():
                    doc_ids[doc_id] = doc_ids.get(doc_id, 0) + position
                    doc_matches_count[doc_id] = doc_matches_count.get(doc_id, 0) + 1

        sorted_doc_ids = sorted(
            doc_ids.items(), key=lambda x: (-doc_matches_count[x[0]], x[1])
        )
        return [
            (self.doc2text[doc_id], doc_matches_count[doc_id], sum_pos)
            for doc_id, sum_pos in sorted_doc_ids
        ]


if __name__ == "__main__":
    titles = pd.read_csv("msu_comments.csv")
    records = titles.to_dict("records")

    tok = PymorphyTokenizer()
    se = SearchEngine(tok)  # без инвертированного индекса

    for record in records:
        se.add_document(record["id"], str(record["text"]))

    print(f'ректор мгу top10 searchEngine_results\n{se.search("ректор мгу")[:10]}')

    sse = SmartSearchEngine(tok)  # с инвертированным индексом

    for record in records:
        sse.add_document(record["id"], str(record["text"]))

    print(f'ректор мгу top10 SmartSearchEngine_results{sse.search("ректор мгу")[:10]}')

    msse = MoreSmartSearchEngine(tok) # с инвертированным индексом 
                                      # и учётом кол-ва совпадений (мб криво работает пока)

    for record in records:
        msse.add_document(record["id"], str(record["text"]))

    print(f'ректор мгу top10 SmartSearchEngine_results{list(zip(*msse.search("ректор мгу")))[0][:10]}')

    #print(msse.doc2text)
