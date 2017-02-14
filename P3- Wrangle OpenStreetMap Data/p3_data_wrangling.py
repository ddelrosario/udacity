
# coding: utf-8

# ## OpenStreetMap project - Manila, Philippines

# In[23]:

# Make a sample OSM file
# http://www.openstreetmap.org/export#map=14/14.5839/121.0153


#!/usr/bin/env python
# -*- coding: utf-8 -*-

import xml.etree.ElementTree as ET  # Use cElementTree or lxml if too slow

OSM_FILE = "manila.osm"  # Replace this with your osm file
SAMPLE_FILE = "manila_main_sample.osm"

k = 10 # Parameter: take every k-th top level element

def get_element(osm_file, tags=('node', 'way', 'relation')):
    """Yield element if it is the right type of tag

    Reference:
    http://stackoverflow.com/questions/3095434/inserting-newlines-in-xml-file-generated-via-xml-etree-elementtree-in-python
    """
    context = iter(ET.iterparse(osm_file, events=('start', 'end')))
    _, root = next(context)
    for event, elem in context:
        if event == 'end' and elem.tag in tags:
            yield elem
            root.clear()


with open(SAMPLE_FILE, 'wb') as output:
    output.write('<?xml version="1.0" encoding="UTF-8"?>\n')
    output.write('<osm>\n  ')

    # Write every kth top level element
    for i, element in enumerate(get_element(OSM_FILE)):
        if i % k == 0:
            output.write(ET.tostring(element, encoding='utf-8'))

    output.write('</osm>')


# In[24]:

# Iterative Parsing

#!/usr/bin/env python
# -*- coding: utf-8 -*-
import xml.etree.cElementTree as ET
import pprint

def count_tags(filename):
    """This function goes through the XML doc and counts the different level tags
    """
    tags = {}
    for event, elem in ET.iterparse(filename):
        if elem.tag in tags:
            tags[elem.tag] += 1
        else:
            tags[elem.tag] = 1
    return tags

def test():

    tags = count_tags('manila.osm')
    pprint.pprint(tags)
    assert tags == { 'bounds' : 1,
                     'member': 11545,
                     'meta' : 1,
                     'nd': 390734,
                     'node': 314738,
                     'note': 1,
                     'osm': 1,
                     'relation': 453,
                     'tag': 133642,
                     'way': 70366}

    

if __name__ == "__main__":
    test()


# In[25]:

# tag types
# Reference
# https://discussions.udacity.com/t/tag-types-and-improving-street-names-locally/46437/20

#!/usr/bin/env python
# -*- coding: utf-8 -*-
import xml.etree.ElementTree as ET
import pprint
import re

lower = re.compile(r'^([a-z]|_)*$')
lower_colon = re.compile(r'^([a-z]|_)*:([a-z]|_)*$')
problemchars = re.compile(r'[=\+/&<>;\'"\?%#$@\,\. \t\r\n]')

def key_type(element, keys):
    """This function goes through the XML doc and examines each tag, <tag>.
    Then it counts the values based on the criteria defined in the variables:
    lower, lower_colon, problemchars.
    """
    # going through the <tag> data
    if element.tag == "tag":
        if 'k' in element.attrib:
            kvalue = element.attrib['k']
            # counting the problemchars
            if problemchars.search(kvalue):
                keys['problemchars'] += 1
            elif lower.match(kvalue):
                # counting the lower          
                keys['lower'] += 1
            elif lower_colon.match(kvalue):
                # counting the lower_colon
                keys['lower_colon'] += 1
            else:
                keys['other'] += 1
    return keys

def process_map(filename):
    keys = {"lower": 0, "lower_colon": 0, "problemchars": 0, "other": 0}
    for _, element in ET.iterparse(filename):
        keys = key_type(element, keys)
    return keys

def test():
    # You can use another testfile 'map.osm' to look at your solution
    # Note that the assertions will be incorrect then.
    keys = process_map('manila.osm')
    pprint.pprint(keys)
    #assert keys == {'lower': 5, 'lower_colon': 0, 'other': 1, 'problemchars': 1}

if __name__ == "__main__":
    test()


# In[26]:

# tag types
# AMENITY

#!/usr/bin/env python
# -*- coding: utf-8 -*-
import xml.etree.cElementTree as ET
from collections import defaultdict
import re

osm_file = open("manila.osm", "r")

amenity_type_re = re.compile(r'\S+\.?$', re.IGNORECASE)
amenity_types = defaultdict(int)

def audit_amenity_type(amenity_types, amenity_name):
    m = amenity_type_re.search(amenity_name)
    if m:
        amenity_type = m.group()

        amenity_types[amenity_type] += 1

def print_sorted_dict(d):
    keys = d.keys()
    keys = sorted(keys, key=lambda s: s.lower())
    # I narrowed the number down
    for k in keys[0:100]:
        v = d[k]
        print "%s: %d" % (k, v) 

def is_amenity_name(elem):
    """This function goes through the XML doc checks for <tag v = "amenity".
    """
    return (elem.tag == "tag") and (elem.attrib['k'] == "amenity")

def audit():
    """This function goes through the XML doc to get the amenity values
    """
    for event, elem in ET.iterparse(osm_file):
        if is_amenity_name(elem):
            audit_amenity_type(amenity_types, elem.attrib['v'])    
    print amenity_types
    #print_sorted_dict(amenity_types)    

if __name__ == '__main__':
    audit()


# In[27]:

# tag types
# viewing the data in tag k =

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import xml.etree.cElementTree as ET
import pprint

def count_ktags(filename):
    """This function goes through the XML doc pulls the <tag k = values
    and counts them.
    """
    ktags = {}
    for event, elem in ET.iterparse(filename):
        for tag in elem.iter("tag"):
            if tag.attrib['k'] in ktags:
                ktags[tag.attrib['k']] += 1
            else:
                ktags[tag.attrib['k']] = 1
    return ktags

def count_vtags(filename):
    """This function goes through the XML doc pulls the <tag v = values
    and counts them.
    """
    vtags = {}
    for event, elem in ET.iterparse(filename):
        for tag in elem.iter("tag"):
            if tag.attrib['v'] in vtags:
                vtags[tag.attrib['v']] += 1
            else:
                vtags[tag.attrib['v']] = 1
    return vtags

def test():

    ktags = count_ktags('manila.osm')
    #pprint.pprint(ktags)
    #print ktags
    
if __name__ == "__main__":
    test()


# In[28]:

# tag types
# viewing the data in tag v =

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import xml.etree.cElementTree as ET
import pprint

def count_ktags(filename):
    """This function goes through the XML doc pulls the <tag k = values
    and counts them.
    """
    
    ktags = {}
    for event, elem in ET.iterparse(filename):
        for tag in elem.iter("tag"):
            if tag.attrib['k'] in ktags:
                ktags[tag.attrib['k']] += 1
            else:
                ktags[tag.attrib['k']] = 1
    return ktags

def count_vtags(filename):
    """This function goes through the XML doc pulls the <tag v = values
    and counts them.
    """
    
    vtags = {}
    for event, elem in ET.iterparse(filename):
        for tag in elem.iter("tag"):
            if tag.attrib['v'] in vtags:
                vtags[tag.attrib['v']] += 1
            else:
                vtags[tag.attrib['v']] = 1
    return vtags

def test():

    vtags = count_vtags('manila.osm')
    #pprint.pprint(vtags)
    #print vtags
    
if __name__ == "__main__":
    test()


# In[29]:

# Exploring Users

#!/usr/bin/env python
# -*- coding: utf-8 -*-
import xml.etree.cElementTree as ET
import pprint
import re

def get_user(element):
    return


def process_map(filename):
    """This function goes through the XML doc and adds the user to a set
    """
    users = set()
    for _, element in ET.iterparse(filename):
        if 'uid' in element.attrib:
            users.add(element.attrib['uid'])
    return users


def test():

    users = process_map('manila.osm')
    print len(users)
    #pprint.pprint(users)
    #print users
    assert len(users) == 760



if __name__ == "__main__":
    test()


# In[30]:

# Improving Street Names
# Used this for viewing data
# Reference
# https://discussions.udacity.com/t/tag-types-and-improving-street-names-locally/46437/20

import xml.etree.cElementTree as ET
from collections import defaultdict
import re
import pprint

#OSMFILE = "example_st.osm"
#OSMFILE = "chicago_sample.osm"
OSMFILE = "manila.osm"
street_type_re = re.compile(r'\b\S+\.?$', re.IGNORECASE)


# updated this set
expected = ["Street", "Avenue", "Boulevard", "Drive", "Court", "Place", "Square", "Lane", "Road", 
            "Trail", "Parkway", "Commons", "Circle"]

# updated this variable from viewing the data
mapping = { "St": "Street",
            "St.": "Street",
            "St.,": "Street",           
            "Sts.": "Street",                      
            "Ave": "Avenue",
            "Ave.": "Avenue",           
            "Rd." : "Road",
            "road" : "Road"
          }


def audit_street_type(street_types, street_name):
    m = street_type_re.search(street_name)
    if m:
        street_type = m.group()
        if street_type not in expected:
            street_types[street_type].add(street_name)


def is_street_name(elem):
    return (elem.attrib['k'] == "addr:street")


def audit(osmfile):
    osm_file = open(osmfile, "r")
    street_types = defaultdict(set)
    for event, elem in ET.iterparse(osm_file, events=("start",)):

        if elem.tag == "node" or elem.tag == "way":
            for tag in elem.iter("tag"):
                if is_street_name(tag):
                    audit_street_type(street_types, tag.attrib['v'])

    return street_types


def update_name(name, mapping):
    """This function goes through the "addr:street" values and updates/standardizes them 
    using the mapping.
    """
    m = street_type_re.search(name)
    if m:
        street_type = m.group()
        if street_type in mapping:
            name = name[0:len(name)-len(street_type)] + mapping[street_type]
    return name


def test():
    st_types = audit(OSMFILE)
    #assert len(st_types) == 3
    #print len(st_types)
    #pprint.pprint(dict(st_types))

    for st_type, ways in st_types.iteritems():
        for name in ways:
            better_name = update_name(name, mapping)
            #print name, "=>", better_name
            if name == "West Lexington St.":
                assert better_name == "West Lexington Street"
            if name == "Baldwin Rd.":
                assert better_name == "Baldwin Road"


if __name__ == '__main__':
    test()


# In[31]:

# Auditing Postal Codes
"""This function goes through the XML doc and for tag k = "addr:postcode",
adds all the values (postal codes) to a set.
"""

import xml.etree.cElementTree as ET
from collections import defaultdict
import re
import pprint

osmfile = "manila.osm"

osm_file = open(osmfile, "r")
postcode_types = defaultdict(set)
    
for event, elem in ET.iterparse(osm_file, events=("start",)):
    if elem.tag == "node" or elem.tag == "way":
        for tag in elem.iter("tag"):
            if tag.attrib['k'] == "addr:postcode":
                postcode_types[tag.attrib['k']].add(tag.attrib['v'])

print postcode_types
#pprint.pprint(dict(postcode_types))


# In[32]:

# Preparing for Database

#!/usr/bin/env python
# -*- coding: utf-8 -*-
import xml.etree.ElementTree as ET
import pprint
import re
import codecs
import json

lower = re.compile(r'^([a-z]|_)*$')
lower_colon = re.compile(r'^([a-z]|_)*:([a-z]|_)*$')
problemchars = re.compile(r'[=\+/&<>;\'"\?%#$@\,\. \t\r\n]')
street_type_re = re.compile(r'\b\S+\.?$', re.IGNORECASE)

CREATED = [ "version", "changeset", "timestamp", "user", "uid"]

# updated this based on viewing data
mapping = { "St": "Street",
            "St.": "Street",
            "St.,": "Street",           
            "Sts.": "Street",                      
            "Ave": "Avenue",
            "Ave.": "Avenue",           
            "Rd." : "Road",
            "road" : "Road"
          }

def update_name(name, mapping):
    """This function goes through the "addr:street" values and updates/standardizes them 
    using the mapping.
    """
    m = street_type_re.search(name)
    if m:
        street_type = m.group()
        if street_type in mapping:
            name = name[0:len(name)-len(street_type)] + mapping[street_type]
    return name

def shape_element(element):
    """This function goes through "node", "way" data and assigns values.
    Then it cleans postal codes and street names.   Finally, it assigns "nd" data.
    """
    node = {}
    # only nodes or ways
    if element.tag == "node" or element.tag == "way" :
        node['type'] = element.tag
        # looping through <node> data and assigning values
        for a in element.attrib:
            if a == "lat":
                if 'pos' not in node:
                    node['pos'] = [0.0, 0.0]
                # preserving decimals
                node['pos'][0] = float(element.attrib['lat'])
            elif a == 'lon':
                if 'pos' not in node:
                    node['pos'] = [0.0, 0.0]
                # preserving decimals
                node['pos'][1] = float(element.attrib['lon'])
            elif a in CREATED:
                if 'created' not in node:
                    node['created'] = {}
                node['created'][a] = element.attrib[a]
            else:
                node[a] = element.attrib[a]
        # looping through <tag> data and assigning values
        for c in element.iter('tag'):
            # assigning the k values
            if 'k' in c.attrib:
                kvalue = c.attrib['k']
                if not problemchars.search(kvalue):
                    if kvalue[0:5] == 'addr:':
                        # cleaning the postal codes to be first 4 digits - some chars in front
                        if kvalue[5:] == 'postcode':
                            try:
                                c.attrib['v'] = re.findall(r'(\d{4})', c.attrib['v'])[0]
                            except IndexError:
                                c.attrib['v'] = 'null'
                        if kvalue.find(':', 5) == -1:
                            if 'address' not in node:
                                node['address'] = {}
                            # encoded & then ran update_name function on street names
                            node['address'][kvalue[5:]] = update_name((c.attrib['v']).encode('utf-8'),mapping)
                    else:
                        node[kvalue] = update_name(c.attrib['v'], mapping)
        # assigning the <nd> data
        for c in element.iter('nd'):
            if 'ref' in c.attrib:
                rvalue = c.attrib['ref']
                if 'node_refs' not in node:
                    node['node_refs'] = []
                node['node_refs'].append(rvalue)
        return node
    else:
        return None


def process_map(file_in, pretty = False):
    # You do not need to change this file
    file_out = "{0}.json".format(file_in)
    data = []
    with codecs.open(file_out, "w") as fo:
        for _, element in ET.iterparse(file_in):
            el = shape_element(element)
            if el:
                data.append(el)
                if pretty:
                    fo.write(json.dumps(el, indent=2)+"\n")
                else:
                    fo.write(json.dumps(el) + "\n")
    return data

def test():
    data = process_map('manila.osm', True)
    #pprint.pprint(data)
    
if __name__ == "__main__":
    test()


# ## MongoDB import

# In[34]:

# Start mongod
# mongoimport --db examples --collection manila_main --file c:\temp\manila.osm.json

# start mongo to query
# show dbs
# use examples


# In[35]:

# Data Wrangling 
# Queries required by rubric and also what I am interested in

def get_db():
    from pymongo import MongoClient
    client = MongoClient('localhost:27017')
    db = client.examples
    return db

db = get_db()

def aggregate(db, pipeline):
    """This function takes the database along with the database query 
    and returns the documents
    """
    return [doc for doc in db.manila_main.aggregate(pipeline)]

users = [{"$group": {"_id":"$source","count":{"$sum":1} } },
        {"$sort":{"count":-1} } ]

postal = [{ "$match" : { "address.postcode" : {"$exists" :1 } } },
         { "$group" : { "_id" : "$address.postcode", "count" : { "$sum" : 1 } } },
         { "$sort" : { "count" : -1 } },
         { "$limit" : 10}]

cities = [{ "$match" : { "address.city" : {"$exists" :1 } } },
         { "$group" : { "_id" : "$address.city", "count" : { "$sum" : 1 } } },
         { "$sort" : { "count" : -1 } },
         { "$limit" : 10}]

unique_user = [{"$group": {"_id":"$created.user","count":{"$sum":1} } },
        {"$sort":{"count":-1} },
        {"$limit": 10}]

once_user = [{"$group": {"_id":"$created.user","count":{"$sum":1} } },
        {"$group": {"_id":"$count","num_users":{"$sum":1} } },     
        {"$sort":{"count":-1} },
        {"$limit": 10}]

amenity = [{ "$match" : { "amenity" : {"$exists" : 1 } } },
         { "$group" : { "_id" : "$amenity", "count" : { "$sum" : 1 } } },
         { "$sort" : { "count" : -1 } },
         { "$limit" : 10}]

religion = [{ "$match" : {"amenity" : {"$exists" : 1}, "amenity" : "place_of_worship" } },
            { "$group" : { "_id" : "$religion", "count" : { "$sum" : 1 } } },
            { "$sort" : {"count" : -1 } }, 
            {"$limit":10 }]

restaurant = [{ "$match" : { "amenity" : {"$exists" : 1 }, 
                        "amenity" : { "$in" : ["fast_food", "restaurant"] } } },
            { "$group" : { "_id" : "$cuisine", "count" : { "$sum" : 1 } } },
            { "$sort" : { "count" : -1 } },
            { "$limit" : 10}]

leisure = [{ "$match" : { "leisure" : {"$exists" :1 } } },
         { "$group" : { "_id" : "$leisure", "count" : { "$sum" : 1 } } },
         { "$sort" : { "count" : -1 } },
         { "$limit" : 10}]

sport = [{ "$match" : { "sport" : {"$exists" :1 } } },
         { "$group" : { "_id" : "$sport", "count" : { "$sum" : 1 } } },
         { "$sort" : { "count" : -1 } },
         { "$limit" : 10}]

streets = [{ "$match" : { "address.street" : {"$exists" :1 } } },
         { "$group" : { "_id" : "$address.street", "count" : { "$sum" : 1 } } },
         { "$sort" : { "count" : -1 } },
         { "$limit" : 10}]

province = [{ "$match" : { "address.province" : {"$exists" :1 } } },
         { "$group" : { "_id" : "$address.province", "count" : { "$sum" : 1 } } },
         { "$sort" : { "count" : -1 } },
         { "$limit" : 10}]

users_result = aggregate(db, users)
postal_result = aggregate(db, postal)
cities_result = aggregate(db, cities)
unique_user_result = aggregate(db, unique_user)
once_user_result = aggregate(db, once_user)
amenity_result = aggregate(db, amenity)
religion_result = aggregate(db, religion)
restaurant_result = aggregate(db, restaurant)
leisure_result = aggregate(db, leisure)
sport_result = aggregate(db, sport)
streets_result = aggregate(db, streets)
province_result = aggregate(db, province)
             
import pprint
print ""
print "User counts"
print users_result
print ""
print "Postal code counts"
print postal_result
print ""
print "City counts"
pprint.pprint(cities_result)
print ""
print "Number of Documents"
print db.manila_main.find().count()
print ""
print "Number of Nodes"
print db.manila_main.find({"type" : "node"}).count()
print ""
print "Number of Ways"
print db.manila_main.find({"type" : "way"}).count()
print ""
print "Number of unique users"
print len(db.manila_main.distinct("created.user"))
print ""
print "Unique user"
pprint.pprint(unique_user_result)
print ""
print "One-time user"
pprint.pprint(once_user_result)
print ""
print "Top 10 amenities"
pprint.pprint(amenity_result)
print ""
print "Top 10 religions"
pprint.pprint(religion_result)
print ""
print "Top 10 restaurants"
pprint.pprint(restaurant_result)
print ""
print "Top 10 leisure"
pprint.pprint(leisure_result)
print ""
print "Top 10 sport"
pprint.pprint(sport_result)
print ""
print "Top 10 streets"
pprint.pprint(streets_result)
print ""
print "Top 10 provinces"
pprint.pprint(province_result)


# In[ ]:



