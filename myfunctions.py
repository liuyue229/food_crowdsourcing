import collections

import pandas as pd
pd.options.mode.chained_assignment = None # default is warn
from pandas.io.json import json_normalize

import flatten_json
import numpy as np

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
    foodpanda = { }
    deliveroo = {}
    wte = {}

    foodpanda.update({"_index":"foodpanda",
                      "food":"menu_item",
                      "vendor":"vendor", "crawling_cycle":"cycle",    
                      "cycle_id":"_source.cycle_id", # from food records     
#                       "food_cycle":'_source.cycle', 
#                       "food_name":"_source.title",
#                       "vendor_name":"_source.vendor.vendor_name", 
#                      "desc":"_source.description",                  
                     "ref":{ 
            '_source.category': 'tag', #food
            '_source.description': 'desc', #food
            '_source.title': 'food_name', #food
            '_source.vendor.vendor_address': 'address',#rest
            '_source.vendor.vendor_name': 'vendor_name',#rest
            '_source.vendor.vendor_rating': 'rating',#rest
            'cuisine': 'restaurant.cuisine',#rest  
            '_source.cycle' : 'timestamp_cycle',
            'price': 'price', #food
            'loc': 'loc'}
                     })
    deliveroo.update({"_index":"deliveroo",
                      "food":"food",
                      "vendor":"restaurant", "crawling_cycle":"cycle",        
                      "cycle_id":"_source.cycle",  # from food records                 
#                       "food_cycle":'_source.restaurant.cycle',
#                       "food_name":'_source.title',
#                       "vendor_name":'_source.restaurant.name',            
                     "ref":{ 
            '_source.category': 'tag', #food
            '_source.description': 'desc', #food
            '_source.title': 'food_name', #food
            '_source.restaurant.address': 'address',#rest
            '_source.restaurant.name': 'vendor_name',#rest
            '_source.restaurant.neighbourhood':"neighbourhood", #rest
            '_source.restaurant.phone': 'phone',#rest
            'opening': 'opening',#rest
            '_source.restaurant.tag': 'restaurant.cuisine',#rest  
            '_source.cycle' : 'timestamp_cycle',         
            'price': 'price', #food
            'loc': 'loc'}})
    
    
    wte.update({"_index":"what_to_eat",
                "food":"food",
                "vendor":"restaurant",
#                 "cycleStart":"_source.startTimestampGMT",
#                 "cycle_id":"_id",
#                 "food_name":'_source.name',
#                 "vendor_name":'_source.restaurant.name',
               "ref" : { 
            '_source.category': 'tag', #food
            '_source.description': 'desc', #food
            '_source.name': 'food_name', #food
            '_source.restaurant.address': 'address',#rest
            '_source.restaurant.name': 'vendor_name',#rest
            '_source.restaurant.rating': 'rating',#rest
            'opening': 'opening',#rest
            'cuisine': 'restaurant.cuisine',#rest     
            '_source.startTimestampGMT': 'timestamp',
            'loc': 'loc',
            'price': 'price', #food
            }})
    return [foodpanda, deliveroo, wte]

def burpple_para():
    # define dict for b and bi
    burpple = {"_index":"burpple"}
    burppleinitial = {"_index":"burppleinitial"}
    burpples = [burpple, burppleinitial]
    for website in burpples:
        website.update({"vendor":"restaurant",
                        "review":"review",  
                        "user":"user",                                   
                        "_id":"_id",
                        "vendor_id":'_source.id',
                        "vendor_url":'_source.url',
                        "vendor_name":"_source.name", 
                        "address":"_source.address", 
                        "cuisine_tags":"_source.tags", 
                        "phone":"_source.phone",                
                        "cycleStart":'_source.crawlTimeStamp'})    
    burpple.update({"reviewFeedTime":'_source.feedDatetime',})
    burppleinitial.update({"reviewFeedTime":'_source.datetime',})
    return burpple, burppleinitial, burpples


# retriving data, returning json objs, for general purposes
def retrive_data(website, doc_type, _print=False):
    # initialise ES
    initialise_es(0)
    
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
    return recs

# retriving data with json_normalize to return df, for small datasets
def retrive_from_es(website, doc_type):
    recs = retrive_data(website, doc_type)
    df = pd.concat([json_normalize(line) for line in recs])
    return df

# convert unicode to proper string
def __if_number_get_string(number):
    converted_str = number
    if isinstance(number, int) or \
            isinstance(number, float):
        converted_str = str(number)
    return converted_str    
def get_unicode(strOrUnicode, encoding='utf-8'):
    strOrUnicode = __if_number_get_string(strOrUnicode)
    if isinstance(strOrUnicode, unicode):
        return strOrUnicode
    return unicode(strOrUnicode, encoding, errors='ignore')
def get_string(strOrUnicode, encoding='utf-8'):
    strOrUnicode = __if_number_get_string(strOrUnicode)
    if isinstance(strOrUnicode, unicode):
        return strOrUnicode.encode(encoding)
    return strOrUnicode 

# flatten json records of some fields
def flatten_json_records(recs, _ref):
    df = pd.DataFrame([flatten_json.flatten(s, ".", root_keys_to_ignore={'_score'}) for s in recs])
    cols = df.columns.tolist()
    # fp: variations.2 (price)
    p_cols = [col for col in cols if "price" in col]
    df["price"] = df[p_cols].mean(1).round(4)
    # wte: restaurant.opening, de: opening
    o_cols = [col for col in cols if "opening" in col]
    df["opening"] = df[o_cols].apply(lambda x : ', '.join([get_string(s) for s in x if s==s]), 1)
    # fp: vendor.vendor_cuisines, wte: restaurant.cuisine
    c_cols = [col for col in cols if "cuisine" in col]
    # df["cuisine"] = df[cuisine_cols].values.tolist()
    df["cuisine"] = df[c_cols].apply(lambda lst : [x for x in lst if x==x], 1)
    # get new col: loc
    df["loc"] = df["_index"] + "/"+df["_type"] +  "/"+df["_id"]
    # map col names & col selection
    df.rename(columns=_ref, inplace=True)
    df = df[_ref.values()] 
    # some foodpanda records contain null values
    df = df[df["food_name"].notnull()] 
    return df

# main function for data retrival and pre-processing
def process_rec(website):
    if 1==1:
        # retrive data
        recs = retrive_data(website, "food")
        # extract relevant attributes
        df = flatten_json_records(recs, website["ref"])
    # append cycle crawling time, do not do parallel
    if website["_index"] != 'what_to_eat':
        d = retrive_from_es(website, "crawling_cycle")
        # create a ref dict for cycle_id: timestamp
        cycle_time = d.set_index(website['cycle_id']).to_dict()["_source.crawlStartDateTimeGMT"] 
        # assign time according to timestamp_cycle
        df["timestamp"] = df['timestamp_cycle'].apply(lambda s: cycle_time[s])
        df = df[[col for col in df.columns.tolist() if col != 'timestamp_cycle']]
    return df

# helper functions for tidying up the records
# get tuple
tup = lambda g: tuple(g)
# get list of unique
unq = lambda g:  sorted(list(set(g)))
# process rating - restaurant
avg_rating = lambda g: np.nanmean(np.array(g).astype(np.float))
# enlist - mixture of list and str, return unique
def enlist(l):
    l = list(tuple(l))
    res = set()
    for i in l:
        if isinstance(i, list):
            res = res | set(i)
        else:
            if i==i:
                res.add(i)
    return sorted(list(res))
# process timestamp - restaurant
def time_range(lst):
    lst = sorted(lst)
    return [lst[0][:10], lst[-1][:10]]

# main function for restaurant entities
def get_restaurant_entities(df):
    fn = {"rating":avg_rating, 
        "restaurant.cuisine":enlist, # for mixture of string and lst, use enlist 
        "loc":tup, # for unq str, use tuple
         "timestamp":time_range,  # time range
         "address":unq, "neighbourhood":unq,"phone":unq, "opening":unq,# for str, use unique
          }
    restaurant = df.groupby("vendor_name").agg(fn)
    # rename column
    restaurant.rename(columns={'timestamp': 'crawling_range'}, inplace=True)
    # free the "name" column
    restaurant.reset_index(inplace=True)
    # update index to be "delivery_"+xxx
    restaurant.reset_index(inplace=True)
    restaurant["index"] = "delivery_"+restaurant["index"].apply(str)
    restaurant.set_index("index", inplace=True)
    return restaurant

# count freq in list, return sorted list of (key, freq) pairs
def count_freq(l):
    from collections import Counter
    c = Counter(l)
    return sorted(c.items(), key=lambda x: -x[1])

# main function for restaurant entities
def get_food_entities(df, restaurant_ref):
    df["vendors"] = df["vendor_name"].apply(lambda s: restaurant_ref[s])
    food_cols = ['food_name', 'vendors','loc', 'timestamp', 'price', 'tag', 'desc']
    # ref dict: food_name: clean_name
    food_names = set(df["food_name"].tolist())
    print "with %d unique menu names in %d records"%(len(food_names), df.shape[0])
    # merge records based on clean name
    clean_food_ref = {s:clean_name_v2(s) for s in food_names}    
    df["clean_name"] = df["food_name"].apply(lambda s: clean_food_ref[s])
    tup = lambda g: tuple(g)
    unq = lambda g:  sorted(list(set(g)))
    fn = {"food_name":count_freq, "price":tup,
          "vendors": unq, "desc":unq, "tag":unq, "loc":unq, 
          "timestamp":unq
         }    
    food_items = df.groupby("clean_name").agg(fn)
    # free the "clean_name" column
    food_items.reset_index(inplace=True)
    # update index to be "food_"+xxx
    food_items.reset_index(inplace=True)
    food_items["index"] = "food_"+food_items["index"].apply(str)
    food_items.set_index("index", inplace=True)
    return food_items

def consolidate_burpple_records(burpples, non_sg_vendors=[], _print=False):
    # not included: '_source.url', # https://www.burpple.com/f/ + "_id"
    cols = ['_id', # review identifier, something like "liKrL-pE"
            '_type',
            '_index', # burpple / burpple initial
            '_source.title', # title of review, with some special characters
            '_source.body', # text 
            '_source.crawlTimeStamp', 
            '_source.foodImgUrl',             
            '_source.username', # user identifier
            '_source.restaurant.id',
            '_source.restaurant.name'] #vendor identifier
    # merge feed time
    import pandas as pd
    pd.options.mode.chained_assignment = None # default is warn
    df = pd.concat([site["all_rec_review"]
                    [cols+ [site["reviewFeedTime"]]] for site in burpples])
    df["feedTime"] = df[[site["reviewFeedTime"] for site in burpples]].fillna('').sum(axis=1)
    df = df[cols +["feedTime"]]
    # sort, only leaving the latest crawled first
    df = df.sort_values(by=['_source.crawlTimeStamp',"_id"], ascending=[False,True])
    df = df.groupby("_id").first()
    df.reset_index(inplace=True)
    non_sg_vendors = non_sg_vendors + ['114803', '119954', '136868', '139058', '149618', 
                                       '155202','156512','16431','165933', '166041', '174029', 
                                       '28318', '43464','51131', '59732', '63212']
    # remove reviews of non-sg vendors
    df = df[~df["_source.restaurant.id"].isin(non_sg_vendors)]
    if _print:
        print ("Got %d unique records of review" % len(df))
        print ("Got %d unique records of images" % 
               df[df['_source.foodImgUrl']!=""]['_source.foodImgUrl'].nunique())
        print ("Got %d unique users" % df['_source.username'].nunique())
        print ("Got %d unique vendors" % df['_source.restaurant.id'].nunique())
        print ("Review time from %s to %s" % (df['feedTime'].min(),df['feedTime'].max()))
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

def clean_review(s):
    """ 
    twitter style text-cleaning, for url, mention, hashtag, 
    also: dollar, score
    not: apostrophe conversion, stop words, emoticons, slang, word standardization
    """
    import re   
    # 1. etract urls (may contain $, #, @), remove url part from s
    p = r'http[s]?://(?:[a-z]|[0-9]|[$-_@#.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+'
    urls = tuple(re.findall(p, s))
    for rep in urls:
        s = s.replace(rep,"")
    # 2. normal cleaning: replace w/ and &
    s = " " + s +" "
    s = s.replace("w/", " with ")
    s = s.replace("&", " and ") 
    # 3. extract hashtags 
    hashtags = tuple(re.findall(r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", s))
    # 4. extract mentions
    mentions = tuple(re.findall(r'(?:@[\w_]+)', s))   
    # 5. extract dolalr amounts
    dollars = tuple([x[0] for x in re.findall(r'(\$\d+([,\.]\d+)?(\+\+)?(\+)?k?)', s)])
    # 6. extract scores
    scores = tuple([x[0] for x in re.findall(r'(\d+([\.]\d+)(\/)\d+([\.]\d+)?)', s)])
    # 7. remove above parts from review, strip and remove multiple space
    words = sorted(list(dollars + scores + hashtags + mentions), key=len, reverse=True)
    for rep in words:
        s = s.replace(rep,"")
    s = re.sub(' +',' ', s.strip()) 
    lst = [s, hashtags, mentions, dollars, scores, urls]
    return (tuple(lst))


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

# flatten list of list
import itertools
def flatten(lst):
    if not isinstance(lst, list):
        lst = [lst]
    if any(isinstance(i, list) for i in lst):
        lst = list(itertools.chain.from_iterable(lst))
    return lst 

def inverse_dict(d1):
    # value type for both dict is list, inverse dictionary
    from collections import defaultdict
    d2 = defaultdict(list)
    for loc, items in d1.items():
        for item in items:
            d2[item].append(loc)
    return d2

# search in a review, list down the possible food names (with attributes) it contain
# longest matching food items
_potential_food_names = []

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

def combi(lst):
    lst = sorted(list(set(lst)))
    index = 1
    pairs = []
    for element1 in lst:
        for element2 in lst[index:]:
            pairs.append([element1, element2])
        index += 1
    return pairs 

def edge_generation(df, edge_type ="food_vendor", print_sample=True, save=True):
    # food restaurant_delivery_boolean
    df.reset_index(inplace=True)
    ref = df.set_index("index").to_dict()["vendors"]
    res = dict()
    index = 0
    for food in df_retrived["index"].tolist():
        for r in sorted(list(set(ref[food]))):
                res.update({"_".join([edge_type,str(index)]):{
                            "index":"_".join([edge_type,str(index)]),
                            "from_id": food, 
                            "weight":1, 
                            "to_id":r, 
                            "type": edge_type}})
                index+=1
    print "number of %s relations found: %d"%(edge_type,index)
    if print_sample:
        from itertools import islice
        print "Sample data: "
        print dict(islice(res.iteritems(), 0, 1))
        print ""
    if save:
        file_name = edge_type+'.json'
        import json
        with open(file_name, 'w') as outfile:
            json.dump(json.dumps(res), outfile)
        print "saved: %s"%file_name
    else:
        return res 


# # detecting parallel stings from food names
# # returning words_to_be_removed, syn_token pairs
# def parallel_detection(names):
#     lst = set()
#     additional_words = []
#     # in case names contain list of length greater than 2
#     name_pairs = flatten([combi(n) for n in names])
#     for name_pair in name_pairs:
#         s1, s2 = name_pair[0].split()+[" "], name_pair[1].split()+ [" "]
#         index_pair = (-1, -1)
#     #     print s1, s2
#         temp = set()
#         for i in range(index_pair[0]+1, len(s1)):
#             for j in range(index_pair[1]+1, len(s2)):
#                 if s1[i] == s2[j]:
#                     if (index_pair[0]+1<i) & (index_pair[1]+1<j):
#                         a = " ".join(s1[index_pair[0]+1:i])
#                         b = " ".join(s2[index_pair[1]+1:j])
#                         lst.add(frozenset([a,b]))
#                         temp = temp | set([a,b])
#                     index_pair = (i,j) 
#         additional_words.append(list((set(s1) | set(s2)) - (set(s1) & set(s2)) - temp))
#     additional_words = count_freq(flatten(additional_words))
#     return additional_words, lst


# should not be used, loading lang module each time when calling the function
# def lemma_name(s):
#     import spacy
#     nlp = spacy.load('en_core_web_sm')
#     return " ".join([token.lemma_ for token in nlp(unicode(s, "utf-8"))])

# Levenstein distance (efficient implementation via numpy), from Wikipedia
# def levenshtein(source, target):
#     import numpy as np
#     if len(source) < len(target):
#         return levenshtein(target, source)

#     # So now we have len(source) >= len(target).
#     if len(target) == 0:
#         return len(source)

#     # We call tuple() to force strings to be used as sequences
#     # ('c', 'a', 't', 's') - numpy uses them as values by default.
#     source = np.array(tuple(source))
#     target = np.array(tuple(target))

#     # We use a dynamic programming algorithm, but with the
#     # added optimization that we only need the last two rows
#     # of the matrix.
#     previous_row = np.arange(target.size + 1)
#     for s in source:
#         # Insertion (target grows longer than source):
#         current_row = previous_row + 1

#         # Substitution or matching:
#         # Target and source items are aligned, and either
#         # are different (cost of 1), or are the same (cost of 0).
#         current_row[1:] = np.minimum(
#                 current_row[1:],
#                 np.add(previous_row[:-1], target != s))

#         # Deletion (target grows shorter than source):
#         current_row[1:] = np.minimum(
#                 current_row[1:],
#                 current_row[0:-1] + 1)

#         previous_row = current_row

#     return previous_row[-1]

# # affine gap distance
# def affinegap(s1,s2):
#     import affinegap
#     return affinegap.normalizedAffineGapDistance(s1,s2)

# def retrive_drinks():
#     # pre-defined lists of drinks
#     import pandas as pd
#     csvfile = "possible_drink.csv"
#     drinks = pd.read_csv(csvfile,header=None).iloc[:,0].tolist()
#     drinks = [""]+[s for s in drinks if s==s]
#     print "number of drinks in drink list: %d"%len(drinks)
#     return drinks


# """ 
# getting each attribute directly
# possible error: some field is nan
# => not used
# """
# import json
# def get_attribute_wte(s):
#     r = {}
#     for k in set(s["_source"].keys()) - set(
#         ['restaurant', 'currency','startTimestampGMT']):
#     #[u'category', u'name', u'price',  u'description'] - set(['restaurant']):
#         r["food_"+k] = s["_source"][k]
#     for k in [u'name', u'cuisine', u'address']:
#         r["restaurant_"+k] = s["_source"]['restaurant'][k]
#     r["opening"] = json.dumps(s["_source"]["restaurant"]["opening"])
#     r['rating'] = round(s["_source"]['restaurant']['rating'],2)
#     r["loc"] = s["_index"] + "/"+s["_type"] +  "/"+s["_id"]
#     r["timestamp"] = s["_source"]['startTimestampGMT']
#     return r

# cnt = 10000
# lst_rec = []
# for s in recs[:cnt]:
#     lst_rec.append(get_attribute_wte(s))
# time: 432 ms