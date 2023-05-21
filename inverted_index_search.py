import pymorphy2
import re
import pandas as pd

from functools import lru_cache

from math import log
from math import floor
import time

from sys import getsizeof, stderr
from itertools import chain
from collections import deque

try:
    from reprlib import repr
except ImportError:
    pass

def total_size(o, handlers={}, verbose=False):
    """ Returns the approximate memory footprint an object and all of its contents.

    Automatically finds the contents of the following builtin containers and
    their subclasses:  tuple, list, deque, dict, set and frozenset.
    To search other containers, add handlers to iterate over their contents:

        handlers = {SomeContainerClass: iter,
                    OtherContainerClass: OtherContainerClass.get_elements}

    """
    dict_handler = lambda d: chain.from_iterable(d.items())
    all_handlers = {tuple: iter,
                    list: iter,
                    deque: iter,
                    dict: dict_handler,
                    set: iter,
                    frozenset: iter,
                   }
    all_handlers.update(handlers)     # user handlers take precedence
    seen = set()                      # track which object id's have already been seen
    default_size = getsizeof(0)       # estimate sizeof object without __sizeof__

    def sizeof(o):
        if id(o) in seen:       # do not double count the same object
            return 0
        seen.add(id(o))
        s = getsizeof(o, default_size)

        if verbose:
            print(s, type(o), repr(o), file=stderr)

        for typ, handler in all_handlers.items():
            if isinstance(o, typ):
                s += sum(map(sizeof, handler(o)))
                break
        return s

    return sizeof(o)




def Binary_Representation_Without_MSB(x):
    binary = "{0:b}".format(int(x))
    binary_without_MSB = binary[1:]
    return binary_without_MSB
 
def EliasGammaEncode(k):
    if (k == 0):
        return '0'
    N = 1 + floor(log(k, 2))
    Unary = (N-1)*'0'+'1'
    return Unary + Binary_Representation_Without_MSB(k)

def EliasGammaDecode(k):
    for idx, digit in enumerate(k):
        if(digit == '1'):
            return k[idx:]
    return -1

def EliasDeltaEncode(x):
    Gamma = EliasGammaEncode(1 + floor(log(x, 2)))
    binary_without_MSB = Binary_Representation_Without_MSB(x)
    return Gamma+binary_without_MSB



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
        self.gamma_encoded_index = []
        self.delta_encoded_index = []

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
            doc_ids.items(), key=lambda x: -doc_matches_count[x[0]]
        )
        return [
            (self.doc2text[doc_id], doc_matches_count[doc_id], sum_pos)
            for doc_id, sum_pos in sorted_doc_ids
        ]


    
if __name__ == "__main__":
    titles = pd.read_csv("msu_comments.csv")
    #print(titles.id.median()) #352635 median id
    records = titles.to_dict("records")

    tok = PymorphyTokenizer()
    #se = SearchEngine(tok)  # без инвертированного индекса


    INT_TO_CHECK = 1342124
    print(f'{INT_TO_CHECK} encoded {EliasGammaEncode(INT_TO_CHECK)}')
    print(f'{INT_TO_CHECK} decoded {int(EliasGammaDecode(EliasGammaEncode(INT_TO_CHECK)),2)}')
    print(bytes(EliasGammaEncode(1342124), 'UTF-8'))

    #for record in records:
    #    se.add_document(record["id"], str(record["text"]))
    #print(f'ректор мгу top10 searchEngine_results\n{se.search("ректор мгу")[:10]}')

    sse = SmartSearchEngine(tok)  # с инвертированным индексом

    #индекс без сжатия:
    
    for record in records:
        sse.add_document(record["id"], str(record["text"]))
    mem_before_encode = total_size(sse.inverted_index)
    
    t1 = time.time_ns()
    for _ in range(100000):
        res  = sse.search("ректор мгу")
    t2 = time.time_ns()
    print(f"Smart search time elapsed {(t2-t1)/100000} nanoseconds\n")

    
    print(f'ректор мгу top10 SmartSearchEngine_results\n{res[:10]}')
    
    del sse
    del res
    
    msse = MoreSmartSearchEngine(tok) # с инвертированным индексом 
                                      # и учётом кол-ва совпадений (мб криво работает пока)
    for record in records:
        msse.add_document(record["id"], str(record["text"]))
    
    t1 = time.time()
    for _ in range(100):
        res = list(zip(*msse.search("ректор мгу")))[0]
    t2 = time.time()
    
    print(f"More smart search time elapsed {(t2-t1)/100} seconds\n")
    print(f'ректор мгу top10 MoreSmartSearchEngine_results{res[:10]}')
    #print(msse.doc2text)
