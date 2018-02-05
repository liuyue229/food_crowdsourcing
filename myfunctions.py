
# coding: utf-8

# ## To retrive data from elastic search

# In[1]:

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


# In[2]:

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
                     "crawling_cycle":"cycle"})
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


# In[3]:

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
    drinks = [""]+drinks
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
    s = ' '.join([w for w in s.split() if len(re.findall('[a-zA-Z]+|\\d+', w))==1]) 
    s = re.sub('[^a-zA-Z\n]', ' ', s) # other character 
    s = ' '.join( [w for w in s.split() if len(w)>1] ) #single character
    s = re.sub(' +',' ', s.strip()) # multiple spaces
    return s

def clean_name_v21(s): 
    import re
    s = clean_name_v1(s)
    s = re.sub(r'[\(\[].*?[\)\]]', ' ', s) # remove parenthesis & contents
    s = clean_name_v2(s)
    return s

def clean_name_v3(s):
    s = clean_name_v21(s).lower()
    return s


def clean_name_v4(s): 
    import re
    s = clean_name_v3(s)
    s = " " + s.replace(" ", "  ") + " "
    words_remove = []
    """ words to remove should be checked """
#     words_remove = retrive_words_remove()
    for w in ["pc", "pcs"] + words_remove:
        s = s.replace(" "+ w.replace(" ","  ") + " ", " ")  #specific words     
    s = re.sub(' +',' ', s.strip()) # multiple spaces
    return s

def today():
    # get date in %Y%m%d format
    import datetime
    return datetime.date.today().strftime('%Y%m%d')





