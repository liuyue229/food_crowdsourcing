
# coding: utf-8

def today():
    # get date in %Y%m%d format
    import datetime
    return datetime.date.today().strftime('%Y%m%d')

# ## To retrive data from elastic search
# set up ElasticSearch object and the URL to access it
def setup_es(isServer):
    # python version
    import sys
    # print "system_info: %s"%sys.version
    # current working directory
    import os
    # print "path_info: %s"%os.getcwd()    
    ## Local on PC/laptop or on VM (10.0.106.122:2)   
    from elasticsearch import Elasticsearch
    port = "9200"
    host = "localhost"
    if isServer:
        host = "10.0.109.54"    
    url = "http://" + host + ":" + port    
    #print "es_info: %s"%url    
    es = Elasticsearch([{'host': host, 'port': port}])
    return es, url

# make sure ES is up and running
def initialise_es(i):
    import requests
    es, url = setup_es(True)
    res = requests.get(url)
    if i:
        print(res.content)


# define dict for the _index, and _doc_type for food and vendor
def delivery_para():
    foodpanda = {}
    deliveroo = {}
    wte = {}
    foodpanda.update({"_index":"foodpanda",
                      "food":"menu_item",
                      "vendor":"vendor",
                      "cycleStart":"_source.crawlStartDateTimeGMT",
                      "cycle_id":"_source.cycle_id",
                      "food_cycle":'_source.cycle', 
                      "food_name":"_source.title",
                      "vendor_name":"_source.vendor.vendor_name",
                     "crawling_cycle":"cycle",
                     "desc":"_source.description"})
    deliveroo.update({"_index":"deliveroo",
                      "food":"food",
                      "vendor":"restaurant",
                      "cycleStart":"_source.crawlStartDateTimeGMT",
                      "cycle_id":"_source.cycle",
                      "food_cycle":'_source.restaurant.cycle',
                      "food_name":'_source.title',
                      "vendor_name":'_source.restaurant.name',
                     "crawling_cycle":"cycle"})
    wte.update({"_index":"what_to_eat",
                "food":"food",
                "vendor":"restaurant",
                "cycleStart":"_source.startTimestampGMT",
                "cycle_id":"_id",
                "food_name":'_source.name',
                "vendor_name":'_source.restaurant.name'})
    return foodpanda, deliveroo, wte, [foodpanda, deliveroo, wte]

# get restaurant entities attributes
def get_restaurant_details(d):
    col_ref = {"address":['_source.restaurant.address', '_source.vendor.vendor_address'],
            "tag":['_source.restaurant.cuisine', '_source.vendor.vendor_cuisines',
            '_source.restaurant.tag'],
            "rating":['_source.restaurant.rating', '_source.vendor.vendor_rating'], 
            "name":['_source.restaurant.name', '_source.vendor.vendor_name'],
            "neighbourhood":['_source.restaurant.neighbourhood'], 
            "opening":['_source.restaurant.opening'],
            "phone":['_source.restaurant.phone']}
    for k,v in col_ref.items():
        d[k] = d[v].apply(lambda x : ''.join([str(s) for s in x if s==s]), axis=1)
    cols = ["loc", "timestamp"] + sorted(col_ref.keys())
    restaurants = d[cols]
    return restaurants 

# get food entities attributes
# get food entities attributes
def get_food_details(d, restaurant_ref):
    dic = {}
    col_ref = {"food_name":["_source.title",'_source.name'],
            "vendor_name":['_source.restaurant.name', '_source.vendor.vendor_name'],
            "desc":["_source.description"],
            "price":['_source.price','_source.variations'],
            "tag":['_source.category']} 
    for k,v in col_ref.items():
        d[k] = d[v].apply(lambda x : ''.join([str(s) for s in x if s==s]), axis=1)
    d["vendors"] = d["vendor_name"].apply(lambda s: restaurant_ref[s])
    cols = ["loc", "timestamp"] + sorted(col_ref.keys())
    cols.remove("vendor_name")
    food_items = d[cols+["vendors"]]   
    return food_items

# retriving data may take a long time
def retrive_from_es(website, doc_type):
    # ES search pattern
    _body = {"query": {"match_all": {}}}
    _index = website['_index']
    _doc_type = website[doc_type]

    # With the help of a generator, get all records
    from elasticsearch import helpers
    es, url = setup_es(True)
    scanResp = helpers.scan(es, _body, scroll= "2m", 
                            index= _index, 
                            doc_type= _doc_type, 
                            timeout="2m")
    recs = []
    for rec in scanResp:
        recs.append(rec)

    # Convert unicode to string (ascii, ignore unicode such as '\xae')
    def convert_unicode(data):
        if isinstance(data, basestring):
            return (data.encode("ascii","ignore"))
        elif isinstance(data, collections.Mapping):
            return dict(map(convert_unicode, data.iteritems()))
        elif isinstance(data, collections.Iterable):
            return type(data)(map(convert_unicode, data))
        else:
            return data

    # json file to dataframe 
    import collections
    import pandas as pd
    pd.options.mode.chained_assignment = None # default is warn
    from pandas.io.json import json_normalize
    lst_rec = []
    for line in recs:
        line = convert_unicode(line)
        lst_rec.append(json_normalize(line))    
    df = pd.concat(lst_rec) 
    # Print shape and all attributes
    #print ("ES location: %s, %s"%(_index,website[doc_type]))
    #print("Dimention: %d , %d"%df.shape)
    #print("Column names::%s"%", ".join(df.columns.tolist()))
    #print ("\n")
    return df


def save_file(df, file_name):
    # save the file as pickle file
    import pickle
    with open(file_name, 'wb') as pfile:
        pickle.dump(df, pfile)
    print "saved: %s"%file_name

def retrive_file(file_name):
    # retrive pickle file
    import pickle   
    with open(file_name, 'rb') as pfile:
        retrived = pickle.load(pfile)
    print ("retrived: %s" % file_name)
    return retrived  

def retrive_drinks():
    # pre-defined lists of drinks
    import pandas as pd
    csvfile = "possible_drink.csv"
    drinks = pd.read_csv(csvfile,header=None).iloc[:,0].tolist()
    drinks = [""]+[s for s in drinks if s==s]
    print "number of drinks in drink list: %d"%len(drinks)
    return drinks

def clean_name_v1(s): 
    try:
        import re
        s = s.replace("\t"," ").replace("\n"," ") 
        s = re.sub(' +',' ', s.strip()) # multiple spaces
        return s
    except:
        return ""

def clean_name_v2(s): 
    import re
    s = clean_name_v1(s)
    s = s.replace("w/o", " no ").replace("W/o", " no ").replace("W/O", " no ").replace("w/O", " no ")
    s = s.replace("w/", " ").replace("W/", " ")
    s = " ".join(re.sub( r"([A-Z])", r" \1", s).split()) #capital letter
    # numeric + character
    #s = ' '.join([w for w in s.split() if len(re.findall('[a-zA-Z]+|\\d+', w))==1]) #wrong 
    #s = re.sub('[^0-9a-zA-Z]+', ' ', s)
    s = re.sub('[^a-zA-Z\n]', ' ', s) # other character 
    s = ' '.join( [w for w in s.split() if len(w)>1] ) #single character
    s = re.sub(' +',' ', s.strip()) # multiple spaces
    s = s.lower()
    return s

def clean_name_v21(s): 
    import re
    s = clean_name_v1(s)
    s = re.sub(r'[\(\[].*?[\)\]]', ' ', s) # remove parenthesis & contents
    s = clean_name_v2(s)
    return s

def clean_name_v3(s):
    s = clean_name_v21(s)
    return s

def clean_name_v4(s): 
    import re
    s = clean_name_v3(s)
    s = " " + s.replace(" ", "  ") + " "
    words_remove = []
    """ not removing words """
#     words_remove = retrive_words_remove() + ["pc", "pcs"]
    for w in words_remove:
        s = s.replace(" "+ w.replace(" ","  ") + " ", " ")  #specific words     
    s = re.sub(' +',' ', s.strip()) # multiple spaces
    return s

# should not be used, loading lang module each time when calling the function
# def lemma_name(s):
#     import spacy
#     nlp = spacy.load('en_core_web_sm')
#     return " ".join([token.lemma_ for token in nlp(unicode(s, "utf-8"))])

# Levenstein distance (efficient implementation via numpy), from Wikipedia
def levenshtein(source, target):
    import numpy as np
    if len(source) < len(target):
        return levenshtein(target, source)

    # So now we have len(source) >= len(target).
    if len(target) == 0:
        return len(source)

    # We call tuple() to force strings to be used as sequences
    # ('c', 'a', 't', 's') - numpy uses them as values by default.
    source = np.array(tuple(source))
    target = np.array(tuple(target))

    # We use a dynamic programming algorithm, but with the
    # added optimization that we only need the last two rows
    # of the matrix.
    previous_row = np.arange(target.size + 1)
    for s in source:
        # Insertion (target grows longer than source):
        current_row = previous_row + 1

        # Substitution or matching:
        # Target and source items are aligned, and either
        # are different (cost of 1), or are the same (cost of 0).
        current_row[1:] = np.minimum(
                current_row[1:],
                np.add(previous_row[:-1], target != s))

        # Deletion (target grows shorter than source):
        current_row[1:] = np.minimum(
                current_row[1:],
                current_row[0:-1] + 1)

        previous_row = current_row

    return previous_row[-1]

# affine gap distance
def affinegap(s1,s2):
    import affinegap
    return affinegap.normalizedAffineGapDistance(s1,s2)

# simhash similar items
def simhash_get_similar(sample_names, _width, _k):
    # return list of list, each sublist is a group of similar items,
    # simhash, get features, "" for NaN and print string for other errors
    def get_simhash_features(s, width=_width):
        import re
        import math
        try:
            s = s.lower()
            s = re.sub(r'[^\w]+', '', s) # ignore empty spaces
            return [s[i:i + width] for i in range(max(len(s) - width + 1, 1))]
        except:
            if not math.isnan(s):
                print s
            return ""
    food_features = [get_simhash_features(s) for s in sample_names]
    # obtain features
    from simhash import Simhash, SimhashIndex
    hash_obj = [Simhash(x) for x in food_features]
    # hash_value = [x.value for x in hash_obj]
    # define a k
    all_obj = {sample_names[i]:hash_obj[i] for i in range(len(sample_names))}
    # allow for _k positions of fingerprints mismatch
    index = SimhashIndex(all_obj.items(), k=_k)
    # get list similar food items
    identified_lsts = []
    for s in sample_names:
        identified_lsts.append(sorted(list(set([s] + index.get_near_dups(all_obj[s])))))
    #filter: containing one or more 
    identified_lsts = [l for l in list(connected_components(identified_lsts)) if len(l)>1]
    return identified_lsts

def simhash_name(df, width=2, k=4, col="clean_name"):    
    if col=="name":
        df["old_name"] = df["name"]
    food_names = [s for s in df[col].tolist() if s!=""]
    # only food of sufficient length
    food_names = [s for s in food_names if len(s.replace(" ", ""))>width]
    sim_lst = simhash_get_similar(food_names, _width=width, _k=k)
    simhash_ref = {l[i]:l[0] for l in sim_lst for i in range(1,len(l))}
    def ref_name(s, ref = simhash_ref):
        try:
            return ref[s]
        except:
            return s
    df["name"] = df[col].apply(ref_name)
    return df

# merge list of list
def connected_components(lists):
    from collections import defaultdict
    neighbors = defaultdict(set)
    seen = set()
    for each in lists:
        for item in each:
            neighbors[item].update(each)
    def component(node, neighbors=neighbors, seen=seen, see=seen.add):
        nodes = set([node])
        next_node = nodes.pop
        while nodes:
            node = next_node()
            see(node)
            nodes |= neighbors[node] - seen
            yield node
    for node in neighbors:
        if node not in seen:
            yield sorted(component(node))

def count_freq(l):
    from collections import Counter
    c = Counter(l)
    return sorted(c.items(), key=lambda x: -x[1])

def flatten(l):
    return [item for sublist in l for item in sublist]

# search in a review, list down the possible food names (with attributes) it contain
# longest matching food items
_potential_food_names = []

import pandas as pd
file_name = "food_items_20180316.csv"
food_items = pd.read_csv(file_name,header=None).iloc[:,0].tolist()
food_items = [s for s in food_items if s==s]
_potential_food_names = food_items

def search_food_v2(searchFor, values = _potential_food_names):
    lst = []
    vs = []
    for key, value in values.iteritems():
        for v in value:
            if " "+v+" " in searchFor:
                lst.append((v, key,value))
                vs.append(v)
    v_unique = longest_unique_entity(vs)
    unique_tuples = [tup for tup in lst if tup[0] in v_unique]   
    return unique_tuples

def search_food(searchFor, values = _potential_food_names):
    lst = []
    vs = []
    for v in values:
        if " "+v+" " in " "+searchFor+" ":
            vs.append(v)
    v_unique = longest_unique_entity(vs) 
    return v_unique

# return "chicken rice" and "fish soup" form ["chicken rice", "chicken", "fish soup", "soup"]
def longest_unique_entity(lst):
    from copy import deepcopy
    lst1 = deepcopy(lst)
    for i in lst:
        for j in lst:
            if (len(i)<len(j)) and (i in j):
                try:
                    lst1.remove(i)
                except:
                    pass
    return(lst1)

# detecting parallel stings from food names
# # returning words_to_be_removed, syn_token pairs
def parallel_detection(names):
    lst = set()
    additional_words = []
    # in case names contain list of length greater than 2
    name_pairs = flatten([combi(n) for n in names])
    for name_pair in name_pairs:
        s1, s2 = name_pair[0].split()+[" "], name_pair[1].split()+ [" "]
        index_pair = (-1, -1)
    #     print s1, s2
        temp = set()
        for i in range(index_pair[0]+1, len(s1)):
            for j in range(index_pair[1]+1, len(s2)):
                if s1[i] == s2[j]:
                    if (index_pair[0]+1<i) & (index_pair[1]+1<j):
                        a = " ".join(s1[index_pair[0]+1:i])
                        b = " ".join(s2[index_pair[1]+1:j])
                        lst.add(frozenset([a,b]))
                        temp = temp | set([a,b])
                    index_pair = (i,j) 
        additional_words.append(list((set(s1) | set(s2)) - (set(s1) & set(s2)) - temp))
    additional_words = count_freq(flatten(additional_words))
    return additional_words, lst

def combi(lst):
    lst = list(set(lst))
    index = 1
    pairs = []
    for element1 in lst:
        for element2 in lst[index:]:
            pairs.append([element1, element2])
        index += 1
    return pairs 

# detecting parallel stings from food names
# # returning syn_token, strict_syn, possible_names
# def parallel_detection(lst): 
#     from collections import OrderedDict
#     # find parallel pair, if "" & token, sorted by count; else, sorted by count
#     # add new string element to list if the element is not in list
#     def add_new(lst, ele):
#         if ele not in lst:
#             lst.append(ele)
#     # construct directed graph
#     def construct_graphs(lst):
#         graph = {}
#         reverse_graph = {}
#         for s in lst:
#             t = s.split(" ")
#             for i in range(len(t)-1):
#                 add_new(graph.setdefault(t[i],list()),t[i+1])
#                 add_new(reverse_graph.setdefault(t[i+1],list()),t[i])              
#         return (graph,reverse_graph)
#     # detect synonyms as from branched graph
#     def detect_syn(graphs):
#         from collections import OrderedDict
#         syn_token = set()
#         for graph in graphs:
#             for key, value in graph.items():
#                 if len(set(value))>1:
#                     syn_token.add(tuple(OrderedDict.fromkeys(value)))
#         return (syn_token) 
#     def detect_strict_syn(graphs):
#         from collections import OrderedDict
#         strict_syn = []    
#         for graph in graphs:
#             syn_token = set()
#             for key, value in graph.items():
#                 if len(set(value))>1:
#                     syn_token.add(tuple(OrderedDict.fromkeys(value)))
#             strict_syn.append(syn_token)    
#         return (set.intersection(*strict_syn))
#     # find all possible starting and ending tokens
#     def dectect_boarder(lst):
#         starts, ends = [], []
#         for s in lst:
#             t = s.split(" ")
#             add_new(starts,t[0])
#             add_new(ends, t[-1])
#         return (starts, ends)
#     # find all possible paths in a graph
#     def find_all_paths(graph, start, end, path=[]):
#             path = path + [start]
#             if start == end:
#                 return [path]
#             if not graph.has_key(start):
#                 return []
#             paths = []
#             for node in graph[start]:
#                 if node not in path:
#                     newpaths = find_all_paths(graph, node, end, path)
#                     for newpath in newpaths:
#                         paths.append(newpath)
#             return paths
#     # names based on paths in the graph with new vertices
#     def enumerate_names(graph, lst):
#         B = []  
#         starts, ends = dectect_boarder(lst)   
#         for start in starts:
#             for end in ends:
#                 B = B + [" ".join(s) for s in find_all_paths(graph, start, end)]
#         return B
#     # make directed graph of words from list
#     graph, reverse_graph = construct_graphs(lst)
#     # find pairs of unigram tokens that are about the same 
#     # linked to the same token from either forward or babckward
#     syn_token = detect_syn([graph, reverse_graph])
#     # linked to the same token from both forward and babckward
#     strict_syn = detect_strict_syn([graph, reverse_graph])
#     # add new edge to the weighted graph
#     for pair in syn_token:
#         for key, values in graph.items():    
#             if set(values) & set(pair):
#                 graph[key] = list(OrderedDict.fromkeys(values + list(pair)))
#     # construct all possible names
#     possible_names = enumerate_names(graph, lst)
#     return syn_token, strict_syn, possible_names


