import pytest
import timeit
from inverted_index_search import *



def test_TokenExistsInIndexedList():
    titles = pd.read_csv("msu_comments.csv")
    records = titles.to_dict("records")[0:2]
    tok = PymorphyTokenizer()
    se = SearchEngine(tok)  

    for record in records:
        se.add_document(record["id"], str(record["text"]))

    listTokenized = list(se.docs[445828])
    wordToSearch = re.split("\W+", records[0]['text'].lower())[0]
    assert tok.tokenize(wordToSearch)[0] in listTokenized

def test_EveryDocumentHasItsOwnRecord():
    titles = pd.read_csv("msu_comments.csv")
    records = titles.to_dict("records")
    tok = PymorphyTokenizer()
    se = SearchEngine(tok)  

    for record in records:
        se.add_document(record["id"], str(record["text"]))

    assert len(se.doc2text) == len(se.docs) == len(records)

def test_timeSpentOnDefaultListSearch_IsMore_ThanOnIndexedListSearch():
    titles = pd.read_csv("msu_comments.csv")
    records = titles.to_dict("records")
    tok = PymorphyTokenizer()
    
    se = SearchEngine(tok)  
    sse = SmartSearchEngine(tok)
    
    for record in records:
        sse.add_document(record["id"], str(record["text"]))    
    t1 = time.time()       
    sse.search("ректор мгу")
    t2 =  time.time()

    for record in records:
        se.add_document(record["id"], str(record["text"]))
    t3 = time.time()
    se.search("ректор мгу")
    t4 = time.time()
    assert (t4-t3 > t2-t1)

@pytest.mark.parametrize("input", ['рано утром', 'зашли', 'гулять', 'политика', 'парламентаризм'])
def test_InvertedIndexIncludesRandomWordFromDocuments(input):
    titles = pd.read_csv("msu_comments.csv")
    records = titles.to_dict("records")
    tok = PymorphyTokenizer()
     
    sse = SmartSearchEngine(tok)
    
    for record in records:
        sse.add_document(record["id"], str(record["text"]))     

    word = tok.tokenize(input)[0]
    assert len(sse.doc2text[list(sse.inverted_index[word])[0]]) > 0

def test_InvertedIndexDoesNotIncludeStrangeWord():
    titles = pd.read_csv("msu_comments.csv")
    records = titles.to_dict("records")
    tok = PymorphyTokenizer()     
    sse = SmartSearchEngine(tok)
    
    for record in records:
        sse.add_document(record["id"], str(record["text"]))     

    assert len(list(sse.inverted_index['daldkaslkdalksdkl'])) == 0    

def test_IndexedListHasCorrectCounter():
    titles = pd.read_csv("msu_comments.csv")
    records = titles.to_dict("records")
    tok = PymorphyTokenizer()
     
    sse = SmartSearchEngine(tok)
    
    for record in records:
        sse.add_document(record["id"], str(record["text"]))    

    doc2text = {}
    inverted_index = defaultdict(set)
    for key in list(sse.doc2text):
        tokens = tok.tokenize(sse.doc2text[key])
        for token in tokens:
            inverted_index[token].add(key)

        doc2text[key] = sse.doc2text[key]            

    counter = 0
    for text in doc2text.values():
        if ('махн' in text):
            counter += 1
    assert counter == len(list(sse.inverted_index['махнуть']))  

@pytest.mark.parametrize("query", ['рано утром', 'зашли', 'гулять', 'политика'])
def test_AllSearchResultsHasSomethingInCommon(query):
    titles = pd.read_csv("msu_comments.csv")
    records = titles.to_dict("records")
    tok = PymorphyTokenizer()
     
    sse = SmartSearchEngine(tok)
    
    for record in records:
        sse.add_document(record["id"], str(record["text"]))    

    searchResult = sse.search(query)        

    intersection = None
    for sentence in searchResult:
        if intersection is None:
            intersection = tok.tokenize(sentence.lower())
            print(intersection)
        else: 
            intersection = [word for word in tok.tokenize(sentence.lower()) if word in intersection]
    assert len(intersection) > 0

@pytest.mark.parametrize("query", ['вышли', 'зашли', 'гулять', 'политика'])
def test_SearchResultHasQueryTokensItself(query):
    titles = pd.read_csv("msu_comments.csv")
    records = titles.to_dict("records")
    tok = PymorphyTokenizer()
     
    sse = SmartSearchEngine(tok)
    
    for record in records:
        sse.add_document(record["id"], str(record["text"]))    

    searchResult = sse.search(query)
    assert tok.tokenize(query.lower())[0] in tok.tokenize(searchResult[0])


def test_MemoryUsedBeforeEncodingIsBiggerThanAfter():
    titles = pd.read_csv("msu_comments.csv")
    records = titles.to_dict("records")
    tok = PymorphyTokenizer()
     
    sse = SmartSearchEngine(tok)
    
    for record in records:
        sse.add_document(record["id"], str(record["text"]))
    mem_before_encode = total_size(sse.inverted_index)
    sse.encode_index_delta()
    mem_after_encode = total_size(sse.inverted_index)    

    assert mem_before_encode > mem_after_encode

def test_DataIsNotLostAfterEncoding():
    titles = pd.read_csv("msu_comments.csv")
    records = titles.to_dict("records")
    tok = PymorphyTokenizer()
     
    sse = SmartSearchEngine(tok)
    
    for record in records:
        sse.add_document(record["id"], str(record["text"]))
    
    before = len(sse.inverted_index)
    sse.encode_index_delta()   
    after = len(sse.inverted_index)
    assert before == after  

@pytest.mark.parametrize("input", ['вышла', 'зашла', 'гулять', 'политика'])
def test_SearchIsStillWorkingAfterEncoding_Decoding(input):
    titles = pd.read_csv("msu_comments.csv")
    records = titles.to_dict("records")
    tok = PymorphyTokenizer()
     
    sse = SmartSearchEngine(tok)
    
    for record in records:
        sse.add_document(record["id"], str(record["text"]))
    before = len(sse.search(input))

    sse.encode_index_delta()   
    sse.decode_all_delta()
    after = len(sse.search(input))

    assert before == after  