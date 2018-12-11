
# coding: utf-8

# ## Segmenting and Clustering Neighborhoods in Toronto

# We will start by importing beautifulsoup4 package in order to build the code to scrape the following Wikipedia page: https://en.wikipedia.org/wiki/List_of_postal_codes_of_Canada:_M.
# 
# Afterwards, we will transform the data into a pandas dataframe performing the below mentioned operations.

# In[1]:



import numpy as np # library to handle data in a vectorized manner

import pandas as pd # library for data analsysis
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

import json # library to handle JSON files

import msgpack

get_ipython().system("conda install -c conda-forge geopy --yes # uncomment this line if you haven't completed the Foursquare API lab")
from geopy.geocoders import Nominatim # convert an address into latitude and longitude values

import requests # library to handle requests
from pandas.io.json import json_normalize # tranform JSON file into a pandas dataframe

# Matplotlib and associated plotting modules
import matplotlib.cm as cm
import matplotlib.colors as colors

# import k-means from clustering stage
from sklearn.cluster import KMeans

#!conda install -c conda-forge folium=0.5.0 --yes # uncomment this line if you haven't completed the Foursquare API lab
get_ipython().system('pip install folium')
import folium # map rendering library

get_ipython().system('conda install -c conda-forge geocoder -y')


print('Libraries imported.')


# In[2]:


# To build the code to scrape the following Wikipedia page
#$ pip install beautifulsoup4
#$ easy_install beautifulsoup4
from bs4 import BeautifulSoup
import requests
url=  requests.get('https://en.wikipedia.org/wiki/List_of_postal_codes_of_Canada:_M')
#with open (url) as html_file:
soup = BeautifulSoup(url.text,'lxml')
#print(soup.prettify())


# In[3]:


import csv
csv_file=open('canada1.csv','w')
csv_writer=csv.writer(csv_file)
csv_writer.writerow(['Postcode', 'Borough', 'Neighbourhood'])
for tr in soup.find_all('tr')[1:]:
    tds = tr.find_all('td')
    if len(tds)==3:
        Postcode=tds[0].text
        Borough=tds[1].text
        Neighbourhood=tds[2].text
    #print(Postcode, Borough, Neighbourhood)
    csv_writer.writerow([Postcode, Borough, Neighbourhood])
csv_file.close()


# In[4]:


import pandas as pd
p_df=pd.read_csv('canada1.csv')
print('Data loaded')


# In[5]:


p_df.head()


# ### 1. Data Cleansing

# In[6]:


p_df[p_df.columns] = p_df.apply(lambda x: x.str.strip('\n'))
p_df.head()


# #### 1.1 Ignore cells with a borough that is "Not assigned"

# In[7]:


df = p_df[p_df.Borough!='Not assigned']
df.head()


# #### 1.2 Combine Neighborhoods belonging to a single Borhough

# In[8]:


df.set_index(['Postcode','Borough'],inplace=True)
res = df.groupby(level=['Postcode','Borough'], sort=False).agg( ','.join)
res.dtypes


# In[9]:


res.reset_index() #to remove set index#


# #### 1.3 How to deal with Borough with "Not assigned" Neighborhood(s)

# In[10]:



res.loc[res['Neighbourhood'] =='Not assigned']=res[res['Neighbourhood'] == 'Not assigned'].index.values[0][1]
res.head()


# In[11]:


df = res.reset_index()


# #### 1.4 What is the number of rows and columns of our Dataframe?

# In[12]:


df.shape


# ## Segmenting and Clustering Neighborhoods in Toronto - Assignment 2
# 
#  #### 2.1 How to get geographical coordinates of each neighborhood.

# In[13]:


get_ipython().system('conda install -c conda-forge geopy --yes ')
from geopy.geocoders import Nominatim 


import matplotlib.cm as cm
import matplotlib.colors as colors

from sklearn.cluster import KMeans

get_ipython().system('conda install -c conda-forge folium=0.5.0 --yes ')
import folium

url="http://cocl.us/Geospatial_data/Geospatial_Coordinates.csv"
coordinates=pd.read_csv(url)
coordinates.columns = ['Postcode', 'Latitude', 'Longitude']
df1 = pd.merge(df,coordinates, on="Postcode")

df1 = df1[df1['Borough'].str.contains('Toronto')].reset_index(drop=True)
df1.head(10)


# In[14]:


# Use geopy library to get the latitude and longitude values of Toronto
address = 'Toronto'

geolocator = Nominatim()
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude
print('The geograpical coordinate of Toronto City are {}, {}.'.format(latitude, longitude))


# In[15]:


#Access to FourSquare

CLIENT_ID = 'KI3TR0QO4JOKMFELOMF3WSOOI3HFNBF5YLW354MYWBKDHEX3' # your Foursquare ID
CLIENT_SECRET = 'QF4ZBLJRBV4BQX52DVWUPEHJ14A2UJABPCZARZQZYTKIISUD' # your Foursquare Secret
VERSION = '20181130' # Foursquare API version

print('Your credentails:')
print('CLIENT_ID: ' + CLIENT_ID)
print('CLIENT_SECRET:' + CLIENT_SECRET)


# In[16]:


# create map of Toronto using latitude and longitude values
map_toronto = folium.Map(location=[latitude, longitude], zoom_start=10)

# add markers to map
for lat, lng, borough, neighborhood in zip(df1['Latitude'], df1['Longitude'], df1['Borough'], df1['Neighbourhood']):
    label = '{}, {}'.format(neighborhood, borough)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7).add_to(map_toronto)  
    
map_toronto


# In[17]:


#get Nearby Values Function
def getNearbyVenues(names, latitudes, longitudes, radius=500):
    LIMIT=100
    venues_list=[]
    for name, lat, lng in zip(names, latitudes, longitudes):
        print(name)
            
        # create the API request URL
        url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            lat, 
            lng, 
            radius, 
            LIMIT)
            
        # make the GET request
        results = requests.get(url).json()["response"]['groups'][0]['items']
        
        # return only relevant information for each nearby venue
        venues_list.append([(
            name, 
            lat, 
            lng, 
            v['venue']['name'], 
            v['venue']['location']['lat'], 
            v['venue']['location']['lng'],  
            v['venue']['categories'][0]['name']) for v in results])

    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
    nearby_venues.columns = ['Neighbourhood', 
                  'Neighbourhood Latitude', 
                  'Neighbourhood Longitude', 
                  'Venue', 
                  'Venue Latitude', 
                  'Venue Longitude', 
                  'Venue Category']
    
    return(nearby_venues)


# In[18]:


toronto_venues = getNearbyVenues(names=df1['Neighbourhood'],
                        latitudes=df1['Latitude'],
                        longitudes=df1['Longitude']
                                  )


# In[19]:


toronto_venues.head()
toronto_venues.columns
toronto_venues.head()


# In[21]:


# Check Count of Venues for Each Neighbourhoo
toronto_venues.groupby('Neighbourhood').count()


# In[22]:


# get the List of Unique Categories
print('There are {} uniques categories.'.format(len(toronto_venues['Venue Category'].unique())))


# In[24]:


# one hot encoding
venues_onehot = pd.get_dummies(toronto_venues[['Venue Category']], prefix="", prefix_sep="")

# add neighborhood column back to dataframe
venues_onehot['Neighborhood'] = toronto_venues['Neighbourhood'] 

# move neighborhood column to the first column
fixed_columns = [venues_onehot.columns[-1]] + list(venues_onehot.columns[:-1])
#fixed_columns
venues_onehot = venues_onehot[fixed_columns]

venues_onehot.head()


# In[25]:


toronto_grouped = venues_onehot.groupby('Neighborhood').mean().reset_index()
toronto_grouped


# In[26]:


toronto_grouped.shape


# In[27]:


num_top_venues = 5

for hood in toronto_grouped['Neighborhood']:
    print("----"+hood+"----")
    temp = toronto_grouped[toronto_grouped['Neighborhood'] == hood].T.reset_index()
    temp.columns = ['venue','freq']
    temp = temp.iloc[1:]
    temp['freq'] = temp['freq'].astype(float)
    temp = temp.round({'freq': 2})
    print(temp.sort_values('freq', ascending=False).reset_index(drop=True).head(num_top_venues))
    print('\n')


# In[28]:


def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    
    return row_categories_sorted.index.values[0:num_top_venues]


# In[29]:


num_top_venues = 10

indicators = ['st', 'nd', 'rd']

# create columns according to number of top venues
columns = ['Neighborhood']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

# create a new dataframe
neighborhoods_venues_sorted = pd.DataFrame(columns=columns)
neighborhoods_venues_sorted['Neighborhood'] = toronto_grouped['Neighborhood']

for ind in np.arange(toronto_grouped.shape[0]):
    neighborhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(toronto_grouped.iloc[ind, :], num_top_venues)

neighborhoods_venues_sorted.head()


# In[30]:


#Distribute in 5 Clusters

# set number of clusters
kclusters = 5

toronto_grouped_clustering = toronto_grouped.drop('Neighborhood', 1)

# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(toronto_grouped_clustering)

# check cluster labels generated for each row in the dataframe
kmeans.labels_[0:50]


# In[31]:



#Dataframe to include Clusters

toronto_merged = df1

# add clustering labels
toronto_merged['Cluster Labels'] = kmeans.labels_

# merge toronto_grouped with toronto_data to add latitude/longitude for each neighborhood
toronto_merged = toronto_merged.join(neighborhoods_venues_sorted.set_index('Neighborhood'), on='Neighbourhood')

toronto_merged.head(30) # check the last columns!


# In[32]:


# Create Map

map_clusters = folium.Map(location=[latitude, longitude], zoom_start=11)

# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i+x+(i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(toronto_merged['Latitude'], toronto_merged['Longitude'], toronto_merged['Neighbourhood'], toronto_merged['Cluster Labels']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7).add_to(map_clusters)
       
map_clusters


# In[33]:


toronto_merged.loc[toronto_merged['Cluster Labels'] == 0, toronto_merged.columns[[1] + list(range(5, toronto_merged.shape[1]))]].head()


# In[34]:


toronto_merged.loc[toronto_merged['Cluster Labels'] == 1, toronto_merged.columns[[1] + list(range(5, toronto_merged.shape[1]))]]


# In[35]:


toronto_merged.loc[toronto_merged['Cluster Labels'] == 2, toronto_merged.columns[[1] + list(range(5, toronto_merged.shape[1]))]]


# In[36]:


toronto_merged.loc[toronto_merged['Cluster Labels'] == 3, toronto_merged.columns[[1] + list(range(5, toronto_merged.shape[1]))]]


# In[37]:


toronto_merged.loc[toronto_merged['Cluster Labels'] == 4, toronto_merged.columns[[1] + list(range(5, toronto_merged.shape[1]))]]

