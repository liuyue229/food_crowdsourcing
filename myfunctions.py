
from elasticsearch import Elasticsearch
from elasticsearch import helpers
import requests

import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None # default is warn
import collections
import itertools
import flatten_json
import copy
import re
import datetime


def today():
    # get date in %Y%m%d format
    return datetime.date.today().strftime('%Y%m%d')

# ## To retrive data from elastic search
# set up ElasticSearch object and the URL to access it
def setup_es(isServer): 
    port = "9200"
    host = "localhost"
    if isServer:
        host = "10.0.109.54"    
    url = "http://" + host + ":" + port     
    es = Elasticsearch([{'host': host, 'port': port}])
    return es, url

# make sure ES is up and running
def initialise_es(i):  
    es, url = setup_es(True)
    res = requests.get(url)
    if i:
        print(res.content)

# define dict for delivey sites
def delivery_para():
    foodpanda = {"_index":"foodpanda",
                "food":"menu_item",
                "vendor":"vendor", "crawling_cycle":"cycle",    
                "cycle_id":"_source.cycle_id", # from food records
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
                }
    deliveroo = {"_index":"deliveroo",
                "food":"food",
                "vendor":"restaurant", "crawling_cycle":"cycle",        
                "cycle_id":"_source.cycle",  # from food records
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
                    'loc': 'loc'}
                 }
    wte = {"_index":"what_to_eat",
            "food":"food",
            "vendor":"restaurant",
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
                }
           }
    return [foodpanda, deliveroo, wte]

# retriving data, returning json objs, for general purposes
def retrive_data(website, doc_type):
    # initialise ES
    initialise_es(0)
    
    # ES search pattern
    _body = {"query": {"match_all": {}}}
    _index = website['_index']
    _doc_type = website[doc_type]

    # With the help of a generator, get all records
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
    df = pd.concat([pd.io.json.json_normalize(line) for line in recs])
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
    res = set()
    l = list(tuple(l))
    for i in l:
        if isinstance(i, list) or isinstance(i, tuple):
            res = res | set(i)
        else:
            if i==i:
                res.add(i)
    return sorted(list(res))
# process timestamp - restaurant
def time_range(lst):
    lst = sorted(lst)
    return [lst[0][:10], lst[-1][:10]]
# process timestamp - food
def time_range_enlist(lst):
    lst = enlist(lst)
    return [lst[0][:10], lst[-1][:10]]
# process price - food
def avg_price(lst):
    return round(np.mean(enlist(lst)),2)

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

# define dict for burpple reviews
def burpple_para():
    burpple = {"_index":"burpple", "reviewFeedTime":'_source.feedDatetime',"review":"review"}
    burppleinitial = {"_index":"burppleinitial", "reviewFeedTime":'_source.datetime',"review":"review"}
    return [burpple, burppleinitial] 

def select_burpple_fields(recs):
    df = pd.DataFrame([flatten_json.flatten(rec, ".", root_keys_to_ignore={'_score'}) for rec in recs])
    # add loc
    df["loc"] = df["_index"] + "/"+df["_type"] +  "/"+df["_id"]
    # select columns
    cols = ['loc', # review identifier, something like burpple/review/liKrL-pE,
            "_id", # image location, liKrL-pE,
            '_source.title', # title of review, with some special characters
            '_source.body', # text 
            '_source.crawlTimeStamp', 
            '_source.foodImgUrl',             
            '_source.username',  #user identifier
            '_source.restaurant.id', #vendor identifier
            '_source.restaurant.name', #vendor name
           ]       
    return df[cols]

def consolidate_burpple_records(dfs, non_sg_vendors=[]):
    # sort, only leaving the latest crawled first
    df = pd.concat(dfs)
    df = df.sort_values(by=['_source.crawlTimeStamp',"_id"], ascending=[False,True])
    df = df.groupby("_id").first()
    df.reset_index(inplace=True)
    non_sg_vendors = non_sg_vendors+ ['114803', '119954', '136868', '139058', '149618', 
                                       '155202','156512','16431','165933', '166041', '174029', 
                                       '28318', '43464','51131', '59732', '63212']
    # remove reviews of non-sg vendors
    df = df[~df["_source.restaurant.id"].isin(non_sg_vendors)]
    return df

# save the file as pickle file
def save_file(df, file_name):
    import pickle
    with open(file_name, 'wb') as pfile:
        pickle.dump(df, pfile)
    print "saved: %s"%file_name

# retrive pickle file
def retrive_file(file_name):
    import pickle   
    with open(file_name, 'rb') as pfile:
        retrived = pickle.load(pfile)
    print ("retrived: %s" % file_name)
    return retrived  

def clean_name_v1(s): 
    try:
        s = s.replace("\t"," ").replace("\n"," ") 
        s = re.sub(' +',' ', s.strip()) # multiple spaces
        return s
    except:
        return ""

def clean_name_v2(s): 
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
        try:
            s = s.lower()
            s = re.sub(r'[^\w]+', '', s) # ignore empty spaces
            return [s[i:i + width] for i in range(max(len(s) - width + 1, 1))]
        except:
            # check nan
            if s==s:
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
    if 1==1:
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
def flatten(lst):
    if not isinstance(lst, list):
        lst = [lst]
    if any(isinstance(i, list) for i in lst):
        lst = list(itertools.chain.from_iterable(lst))
    return lst 

# inverse of a dict
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
def search_food(searchFor, values):
    lst = []
    vs = []
    for v in values:
        if " "+v+" " in " "+searchFor+" ":
            vs.append(v)
    v_unique = longest_unique_entity(vs) 
    return v_unique

# return "chicken rice" and "fish soup" form ["chicken rice", "chicken", "fish soup", "soup"]
def longest_unique_entity(lst):
    lst1 = copy.deepcopy(lst)
    for i in lst:
        for j in lst:
            if (len(i)<len(j)) and (i in j):
                try:
                    lst1.remove(i)
                except:
                    pass
    return(lst1)

# pairs from list
def combi(lst):
    lst = sorted(list(set(lst)))
    index = 1
    pairs = []
    for element1 in lst:
        for element2 in lst[index:]:
            pairs.append([element1, element2])
        index += 1
    return pairs