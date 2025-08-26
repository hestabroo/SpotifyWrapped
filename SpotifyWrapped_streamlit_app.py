#!/usr/bin/env python
# coding: utf-8

# In[154]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import streamlit as st
import time

# In[155]:

st.title("Spotify Wrapped+ 🎧")
st.write("An all-time listening analysis tool by Hayden Estabrook.")


st.write("")
st.subheader("File Upload")
import zipfile, glob, json

zipobj = st.file_uploader(
    "Upload your Spotify Extended Play History (.zip format).  Don't have your play history data yet? [Click here](https://www.spotify.com/us/account/privacy/) to download it!", 
    type=['zip']
)
while zipobj is None:
    st.stop()  #wait until we have a file

st_progress_text = st.empty()
st_progress_bar = st.progress(0)


st_progress_text.write("🤐 Un-zipping your data...")
with zipfile.ZipFile(zipobj) as z:
    files = [f for f in z.namelist() if f.startswith("Spotify Extended Streaming History/Streaming_History_Audio") and f.endswith(".json")]
    files = sorted(files)

    dfs=[]
    for f in files:
        with z.open(f) as fo:
            data = pd.DataFrame(json.load(fo))
            dfs.append(data)

streamhx = pd.concat(dfs).reset_index()


#helpful stuff
st_progress_text.text("🧼 Cleaning up formatting...")
st_progress_bar.progress(1)
time.sleep(1)  #make this loading bar seem cool

streamhx = streamhx[streamhx['audiobook_title'].isna()]  #remove audiobooks

streamhx['dttm'] = pd.to_datetime(streamhx['ts'])
streamhx['dttm_local'] = streamhx['dttm'].dt.tz_convert('America/New_York')  #convert to local timezone

streamhx['year'] = streamhx['dttm'].dt.year
streamhx['month_start'] = streamhx['dttm'].dt.to_period("M").dt.start_time
streamhx['week_start'] = streamhx['dttm'].dt.to_period("W").dt.start_time
streamhx['hour'] = streamhx['dttm'].dt.hour
streamhx['weekday'] = streamhx['dttm'].dt.day_name()

streamhx['hr_played'] = streamhx['ms_played'] / 1000 / 60 / 60

streamhx.rename(columns={
    'master_metadata_track_name':'song_name',
    'master_metadata_album_artist_name':'artist_name',
    'master_metadata_album_album_name': 'album_name'
}, inplace=True)  #simplify some column names


# In[158]:


#basic stuff, top artists
st_progress_text.text("📈 Analyzing top tracks and artists...")
st_progress_bar.progress(5)
time.sleep(1)  #make this loading bar seem cool

artists = streamhx.groupby(by='artist_name').agg(
    hr_played = ('hr_played', 'sum'),
    ct = ('ts', 'count')
)

artists = artists.sort_values(by='hr_played', ascending=False)
artists = artists.head(10)  #we only care about the top 10

#for kicks, could we identify the time range each artist was generally most active?
artist_month = streamhx[streamhx['artist_name'].isin(artists.index)].groupby(by=['artist_name', 'month_start'], as_index=False)['hr_played'].sum()
artist_month['artist_total'] = artist_month['artist_name'].map(artists['hr_played'])
artist_month['pct_total'] = artist_month['hr_played'] / artist_month['artist_total']

#find the smallest possible window containing x% of play time
artist_month.sort_values(by=['artist_name', 'month_start'], inplace=True)
target = 0.5

top_ranges = []
for a in artists.index:  #per artist...
    _df = artist_month[artist_month['artist_name']==a].reset_index()
    _best = {'start': 0, 'end': len(_df)-1, 'size':len(_df), 'pct':1.0}  #initialize the "score to beat" as the whole thing

    for _start in range(len(_df)-1):  #for each possible starting point...
        for _end in range(_start, len(_df)-1):  #slide the window right until...
            _pct = _df['pct_total'][_start:_end+1].sum()
            if _pct >= target:  #...until we capture 50% of volume
                _size = _end-_start
                if _size < _best['size'] or (_size == _best['size'] and _pct > _best['pct']):  #if this is smaller, override best (break ties on higher pct
                    _best = {
                        'start': _start,
                        'end': _end,
                        'size': _size,
                        'pct': _pct
                    }
                break  #and go to the next _start

    #at the end, pull the best start and end combo...
    _startdt = _df['month_start'][_best['start']]
    _enddt = _df['month_start'][_best['end']]

    top_ranges.append(f"{_startdt:%b '%y} - {_enddt:%b '%y}")


artists['peak_range'] = top_ranges

df_artists_display = artists.reset_index()[['artist_name', 'ct', 'hr_played', 'peak_range']].rename(columns={
    'artist_name': "Artist",
    'ct': "Play Count",
    'hr_played': "Total Hours Listened",
    'peak_range': "Most Listened Between"
})
df_artists_display.index += 1
df_artists_display = df_artists_display.style.format({
    "Play Count": '{:,}',
    "Total Hours Listened": '{:.0f}'
})



# In[159]:


#try plotly for interactive
import plotly.express as px
artist_month['hrs_perweek'] = artist_month['hr_played'] / 4.3  #avg weeks/month

artist_month['artist_name'] = pd.Categorical(artist_month['artist_name'], categories=artists.index, ordered=True)  #convert to categorical to allow sorting by top artists
artist_month = artist_month.sort_values(['artist_name', 'month_start'])  #sort (required upfront for accurate rolling)

artist_month['6mo_avg'] = (  #calculate a 6mo avg
    artist_month.groupby('artist_name', observed=False)['hrs_perweek'].transform(lambda x: x.rolling(window=6).mean())
)

px_artisttrend = px.area(
    artist_month,
    x='month_start',
    y='6mo_avg',
    color='artist_name',
    line_shape='spline',
    labels = {
        'month_start': 'Month',
        'artist_name': 'Artist',
        '6mo_avg': 'Weekly Hours'
    },
    hover_data={'6mo_avg':':.1f'},  #set formatting for hover
    title="All-Time Listening to Top Artists"
)

px_artisttrend.update_traces(line_width=0.5)
px_artisttrend.update_layout(plot_bgcolor='white', xaxis_title='')

for f in [px_artisttrend.update_xaxes, px_artisttrend.update_yaxes]:  #iteratiely update both axis
    f(gridcolor='gainsboro', griddash='dot', gridwidth=0)
    



# # Top Songs
songs = streamhx.groupby(by=['song_name', 'artist_name'], as_index=False).agg(
    total_hrs=('hr_played', 'sum'),
    times_played=('ts','count')
)

songs = songs.sort_values(by='times_played', ascending=False).reset_index(drop=True)

#biggest week
songweeks = streamhx.groupby(by=['song_name', 'artist_name', 'week_start'], as_index=False).agg(
    times_played=('ts','count'),
    total_hrs=('hr_played','sum')
)

_topids = songweeks.groupby(by=['song_name', 'artist_name'])['times_played'].idxmax()
best_weeks = songweeks.loc[_topids]
best_weeks.rename(columns={
    'week_start': 'top_week',
    'times_played': 'plays_top_week',
    'total_hrs': 'hrs_top_week'
}, inplace=True)
songs = songs.merge(best_weeks, on=['song_name', 'artist_name'])

df_topsongs_display = songs[['song_name', 'artist_name', 'times_played', 'total_hrs', 'top_week']].head(50).rename(columns={
    'song_name': 'Track',
    'artist_name': 'Artist',
    'times_played': 'Play Count',
    'total_hrs': 'Total Hours Listened',
    'top_week': 'Top Listening Week'
})
df_topsongs_display.index = df_topsongs_display.index + 1  #better rank
df_topsongs_display = df_topsongs_display.style.format({
    "Play Count": '{:,}',
    "Total Hours Listened": '{:.1f}',
    "Top Listening Week": '{:%b %d, %Y}'
})



# In[161]:


#what were your obsessions??
threshold = np.percentile(songweeks.groupby(by='week_start')['times_played'].max(), 95)  #make this dynamic to different listening styles
obsessions = songweeks[(songweeks['times_played']>=threshold) & (songweeks['total_hrs']>threshold*1/60)]  #there seem to be weird weeks with lots of very short plays... exclude

#just take the top 1 song per week
_topids = obsessions.groupby('week_start')['times_played'].idxmax()
obsessions = obsessions.loc[_topids]
obsessions.sort_values(by='week_start', inplace=True)

df_obsessions_display = obsessions.reset_index()[['week_start', 'song_name', 'artist_name', 'times_played', 'total_hrs']].rename(columns={
    'song_name': "Track", 
    'artist_name': "Artist", 
    'week_start': "Week Of", 
    'times_played': "Times Played", 
    'total_hrs': "Total Hours"
})
df_obsessions_display.index += 1
df_obsessions_display = df_obsessions_display.style.format({
    "Week Of": '{:%b %d, %Y}',
    "Times Played": '{:,}',
    "Total Hours": '{:.1f}'
})



# # Listening Times & Patterns

# In[201]:
st_progress_text.text("🔍 Identifying listening patterns...")
st_progress_bar.progress(10)
time.sleep(1)  #make this loading bar seem cool

from datetime import timedelta

weekly = streamhx.groupby(by='week_start', as_index=False)['hr_played'].sum()
weekly = weekly.sort_values('week_start')
weekly['6mo_avg'] = weekly['hr_played'].rolling(window=26).mean()

#find peak 12mo
best = {'start':0, 'end':0, 'hours':0}  #initialize
for _start in range(len(weekly)-51):
    _end = _start+51  #12mo window
    _hours = weekly.iloc[_start:_end]['hr_played'].sum()
    if _hours > best['hours']:
        best = {'start':_start, 'end':_end, 'hours':_hours}

best['startdt'] = weekly.iloc[best['start']]['week_start']
best['enddt'] = weekly.iloc[best['end']]['week_start'] + timedelta(days=6)  #end of week, not start


#last year
lastyr = weekly.iloc[-52:]['hr_played'].sum()
pct_change = (lastyr - best['hours']) / best['hours']



# In[205]:


#maybe plotly
import plotly.express as px

_peakyn = [True if best['start'] <= x <= best['end'] else False for x in weekly.index]
colors = ['Peak' if x else 'Regular' for x in _peakyn]

px_totalhrstrend = px.bar(
    weekly,
    x='week_start',
    y='hr_played',
    color=colors,
    color_discrete_sequence = ['darkgrey', 'limegreen'],
    labels = {
        'week_start': 'Week Starting',
        'color': 'Period',
        'hr_played': 'Weekly Hours'
    },
    hover_data={'hr_played':':.1f'},  #set formatting for hover    
    title="All-Time Hours Listened"
)

px_totalhrstrend.add_scatter(
    x=weekly['week_start'],
    y=weekly['6mo_avg'],
    mode='lines',
    line={'color':'dimgrey', 'dash':'dot', 'width':1.5},
    name='6mo Avg.'
)

px_totalhrstrend.update_layout(plot_bgcolor='white', xaxis_title='')

for f in [px_totalhrstrend.update_xaxes, px_totalhrstrend.update_yaxes]:  #iteratiely update both axis
    f(gridcolor='gainsboro', griddash='dot', gridwidth=0)




import matplotlib.patches as patches

start_date = np.percentile(streamhx['dttm'],1)  #exclude tail before really using account...if like me
timedata = streamhx[streamhx['dttm']>=start_date].copy()
datadays = (timedata['dttm'].max() - timedata['dttm'].min()).days

timelabels = {}
for _ in range (24):  #quick lookup for time formatting
    _hr = (_%12) or 12
    timelabels[_] = f"{_hr}:00 {'AM' if _<12 else 'PM'}"

hmap = timedata.pivot_table(index='hour', columns='weekday', values='hr_played', aggfunc = lambda x: x.sum()/(datadays/7))
hmap.index = hmap.index.map(timelabels)  #am/pm time values
hmap = hmap[['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday' ,'Saturday', 'Sunday']]  #weekdays in order

f_hrsheatmap, ax = plt.subplots(figsize=(8,6))
sns.heatmap(hmap, annot=False, fmt='.0%', cmap = 'coolwarm', annot_kws={'size':8}, ax=ax, vmin=0)
ax.set_xlabel('')
ax.set_ylabel('')
ax.set_title("Average Hourly Listening")

#define these patches so we can reuse them later
def wkday_patch():
    return patches.Rectangle(  #workday
        xy=(0,17),
        width=7,
        height=-8,
        fill=False,
        edgecolor='dimgrey',
        lw=1
    )

def wkend_patch():
    return patches.Rectangle(  #weekend
        xy=(5,24),
        width=2,
        height=-24,
        fill=False,
        edgecolor='dimgrey',
        lw=1
    )

def border_patch():
    return patches.Rectangle(  #full border
        xy=(0,24),
        width=7,
        height=-24,
        fill=False,
        edgecolor='dimgrey',
        lw=1
    )

ax.add_patch(wkday_patch())
ax.add_patch(wkend_patch())
ax.add_patch(border_patch())

cbar = ax.collections[0].colorbar  #override with generic labels, don't get into explaining zscores
cbar.set_ticks([0, np.max(hmap)])
cbar.set_ticklabels(["No Music", "Lots of Music!"])



# In[206]:


import calendar

timedata['month_fmt'] = timedata['month_start'].dt.strftime('%B')

hmap=timedata.pivot_table(index='year', columns='month_fmt', values='hr_played', aggfunc = lambda x: x.sum()/4.3)  #weekly hours
hmap = hmap[calendar.month_name[1:]].sort_index()  #sort months & years

calendarheatmap, ax = plt.subplots(figsize=(12,4))
sns.heatmap(hmap, annot=True, fmt='.1f', cmap='coolwarm', annot_kws={'size':8}, ax=ax)



# # Platform Used

# In[166]:


platform = streamhx.groupby(by=['month_start', 'platform'], as_index=False)['hr_played'].sum()
platform['weekly_hrs'] = platform['hr_played']/4.3

map_terms = {
    'android': 'Android',
    'ios': 'iPhone',
    'os x': 'Mac',
    'windows': 'Windows PC',
    'sonos': 'Sonos',
    'roku': 'Roku'
}

platform['platform_cat'] = platform['platform'].map(lambda x: next((v for k, v in map_terms.items() if k in x.lower()), 'Other'))

platformpie = px.pie(
    platform,
    names = 'platform_cat',
    values='hr_played'
)




# In[190]:


#fill in missing values, causing spikes per platform
months = platform['month_start'].unique()

_list = []
for p in platform['platform_cat'].unique():
    _df = platform[platform['platform_cat']==p].groupby('month_start', as_index=False)['weekly_hrs'].sum()
    _full = pd.DataFrame(months, columns=['month_start'])
    _full = _full.merge(_df, how='left', on='month_start')
    _full['platform_cat'] = p
    _list.append(_full)

platform_full = pd.concat(_list, ignore_index=True)
platform_full = platform_full.fillna(0)

_cat_order = platform.groupby('platform_cat')['hr_played'].sum().sort_values(ascending=False).index
platform_full['platform_cat'] = pd.Categorical(platform_full['platform_cat'], categories=_cat_order, ordered=True)

platform_full = platform_full.sort_values(['platform_cat', 'month_start'])
platform_full['6mo_avg'] = (
    platform_full.groupby('platform_cat', observed=False)['weekly_hrs'].transform(lambda x: x.rolling(window=6).mean())
)

platformtrend=px.area(
    platform_full,
    x='month_start',
    y='6mo_avg',
    color='platform_cat',
    line_shape = 'spline'
)

platformtrend.update_traces(line_width=0.5)

platformtrend.update_layout(plot_bgcolor='white', xaxis_title='')

for f in [platformtrend.update_xaxes, platformtrend.update_yaxes]:  #iteratiely update both axis
    f(gridcolor='gainsboro', griddash='dot', gridwidth=0)


# # Metadta and Sentiment Analysis

# In[168]:


#DAMM - spotify's audio features API has been disabled for free users... pivot to last.fm
#last fm data is much more populated by ARTIST, not track
#**limit to the artists making up x% of playtime?  better trends and save the API
st_progress_text.text("👨‍🎤 Finding your top styles...")
st_progress_bar.progress(15)
time.sleep(2)  #make this loading bar seem cool

_artrank = streamhx.groupby('artist_name')['hr_played'].sum().sort_values(ascending=False)
_cumpct = _artrank.cumsum() / _artrank.sum()
topartists = _cumpct[_cumpct<=0.90].index


# In[169]:


#call last.fm for artist genre tags
import requests

API_key = 'cf00a9af832f39ca55ec767b551ec067'
url = 'http://ws.audioscrobbler.com/2.0/'

params = {
    'api_key': API_key,
    'method': 'artist.getTopTags',
    'format': 'json'
}


#just for testing, see if we can grab from local dir to save from hammering the API
import pickle
try:
    _ppath = r"/Users/haydenestabrook/Documents/Data Science Personal Portfolio/20250811 Spotify Wrapped/lastfm_artistTopTags.pkl"
    with open(_ppath, 'rb') as f:
        responses = pickle.load(f)
    responses = {k: responses[k] for k in responses if k in topartists}  #only keep ones currently in scope
except: responses = {}


try:  #wrapping this whole thing in a try/except to handle the case of API shutdowm
    _c = 0  #tracking
    for a in topartists[~topartists.isin(list(responses))]:  #only run things we haven't got back yet
        params['artist'] = a
        _resp = requests.get(url, params)
        responses[a] = _resp.json()

        if _c%1==0:
            st_progress_text.text(f"📡 Gathering user tags from Last.fm... ({_c/len(topartists):.1%})")
            st_progress_bar.progress(int(15 + 65*_c/len(topartists)))
        _c+=1
except:
    st.error("Sorry, it looks like the Last.fm API is too busy to execute right now...  Please try again in a few minutes.")
    st.stop()


# In[170]:
st_progress_text.text("♻️ Cleaning up tags...")
st_progress_bar.progress(80)
time.sleep(2)  #make this loading bar seem cool

#the artistTopTags reponse already standardizes tags so that top tag = 100 and others are scaled :)


import string

responses = {artist: data for artist, data in responses.items() if 'toptags' in data}  #exclude responses without the toptags key (rare not found errors)

artist_tags_long = []                              
for artist, data in responses.items():  #blow out dictionaries to long list of features
    for tag in data['toptags']['tag'][:]:  #only take the top x tags? [:x]
        _tagname = tag['name']
        _ct = tag['count']
        artist_tags_long.append({'artist_name':artist, 'tag_name':_tagname, 'ct':_ct})  #only the top 3 tags

artist_tags_long = pd.DataFrame(artist_tags_long)  #convert to df
artist_tags_long['tag_name'] = artist_tags_long['tag_name'].str.title()  #convert to title case for better matching
artist_tags_long['tag_name'] = artist_tags_long['tag_name'].str.replace(f"[{string.punctuation}]", " ", regex=True)
artist_tags_long = artist_tags_long[~artist_tags_long['tag_name'].isin(['All', 'Canadian', 'Canada', 'Usa', 'American'])]  #get rid of that garbage "All" tag

#cut out noisy tags
tags_all = artist_tags_long.groupby('tag_name')['ct'].sum()  #cut down noisy tags
tags_keep = tags_all.sort_values(ascending=False).head(50).index  #top 100 only
artist_tags_long = artist_tags_long[artist_tags_long['tag_name'].isin(tags_keep)] 

artist_tags = artist_tags_long.pivot_table(index='artist_name', columns='tag_name', values='ct', aggfunc='max')  #convert to wide format  #for some reason, some artists have the same tag twice... use aggfunc=max
artist_tags = artist_tags.fillna(0)

# In[ ]:





# ## Clustering (KMeans, (H)DBScan)

# In[171]:
st_progress_text.text("👯‍♀️ Deriving optimum genre clustering...")
st_progress_bar.progress(85)
time.sleep(5)  #make this loading bar seem cool

#let's kmeans cluster!  First, elbow method to find the ideal n
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler  #options
from sklearn.decomposition import PCA, TruncatedSVD  #options
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

#scale across ARTISTS, not features?
scaled_tags = StandardScaler().fit_transform(artist_tags.T).T
artist_tags = pd.DataFrame(  #keep the index and column names
    scaled_tags,
    columns = artist_tags.columns,
    index = artist_tags.index
)


results = {}
_nrange = range(15,31)  #max 10-30 clusters
for n in _nrange:
    kmodel = Pipeline([
        ('scaler', StandardScaler()),
        #('pca', PCA(n_components=int(artist_tags.shape[1]/3))),
        ('kmeans', KMeans(n_clusters=n, random_state=69, n_init=100))
    ])
    kmodel.fit(artist_tags)  #don't need to drop artist_name because it's the index

    _inertia = kmodel.named_steps['kmeans'].inertia_
    _silhouette = silhouette_score(artist_tags, kmodel.named_steps['kmeans'].labels_)
    _DBI = davies_bouldin_score(artist_tags, kmodel.named_steps['kmeans'].labels_)

    results[n] = {
        'inertia':_inertia, 
        'silhouette':_silhouette, 
        'DBI':_DBI,
        'labels': kmodel.named_steps['kmeans'].labels_,
        'model': kmodel
    }
    
    if n%10==0: print(f"{n}/{max(_nrange)} complete")

#find the optimum n
dbis = pd.Series({key: val['DBI'] for key, val in results.items()})
dbi_roll = dbis.rolling(window=3).mean()
n = dbis.idxmin()  #could probably just say n=30... always ends up around there

n = 15  #forcing 15 and just taking top 90% seems to give most interpretable results... if confirmed can remove the loop above

print(f"\nn={n}\n\nInertia: {results[n]['inertia']:.3f}\nDBI: {results[n]['DBI']:.3f}\n")

#trainingperf, ax = plt.subplots()
#pd.DataFrame.from_dict(results, orient='index').plot(y=['inertia', 'silhouette', 'DBI'], secondary_y=['inertia'], ax=ax)
#st.pyplot(trainingperf)  #don't print this

# In[172]:


st_progress_text.text("📝 Condensing and naming clusters...")
st_progress_bar.progress(95)
time.sleep(1)  #make this loading bar seem cool


kmodel = results[n]['model']  #retrieve the optimum model

clusters = pd.DataFrame(
    kmodel.named_steps['kmeans'].cluster_centers_,
    columns = artist_tags.columns
)

#clustermap, ax = plt.subplots(figsize=(14,10))
#sns.heatmap(clusters.T, annot=True, fmt='.2f', annot_kws={'size':5}, ax=ax)
#st.pyplot(clustermap)   #don't print this


#add names back into the artists
naming_threshold = np.mean(clusters.max())/5  #could be different for different data
cnames = {}
for _c, r in clusters.iterrows():
    r = r.where(r>naming_threshold).dropna()  #only positive values
    _top2 = r.sort_values(ascending=False).index[:2]
    _name = '/'.join(_top2) if len(_top2)>0 else 'Other'

    cnames[_c] = _name

tagged_artists = pd.DataFrame({
    'artist_name': artist_tags.index,
    'cluster_num': kmodel.named_steps['kmeans'].labels_,
    'cluster_name': pd.Series(kmodel.named_steps['kmeans'].labels_).map(cnames)
})

tagged_artists['hr_played'] = tagged_artists['artist_name'].map(streamhx.groupby(by='artist_name')['hr_played'].sum())


# In[174]:


#put labels back into streamhx
artist_labels = tagged_artists.set_index('artist_name')['cluster_name']
streamhx['cluster_name'] = streamhx['artist_name'].map(artist_labels)

_crank = streamhx.groupby('cluster_name')['hr_played'].sum().sort_values(ascending=False)
_cumpct = _crank.cumsum() / _crank.sum()
graph_clusters = _cumpct[_cumpct<=0.90].index  #only include categories that explain 90% of listening
graph_clusters = [c for c in graph_clusters if c != 'Other']  #don't bother showing that "other" either if it shows up


# In[175]:
st_progress_text.text("📊 Compiling final visualizations...")
st_progress_bar.progress(99)
time.sleep(2)  #make this loading bar seem cool

#add a straight up pie chart
_graph = streamhx[streamhx['cluster_name'].isin(graph_clusters)]

_colors = sns.color_palette('Paired', len(graph_clusters)).as_hex()

px_clusterpie = px.pie(
    names=_graph['cluster_name'],
    values=_graph['hr_played'],
    color_discrete_sequence=_colors,
    title="My Musical Style"    
)
px_clusterpie.update_traces(texttemplate='%{percent:.0%}')  #no decimals



_grp = streamhx.groupby(by=['month_start', 'cluster_name'], as_index=False)['hr_played'].sum()

#for smooth 6mo avgs, will need to fill in missing months per cluster
months = streamhx['month_start'].unique()
clusters = graph_clusters

full_index = [[c, m] for c in clusters for m in months]
cluster_months = pd.DataFrame(full_index, columns=['cluster_name', 'month_start'])
cluster_months = cluster_months.merge(_grp, on=['cluster_name', 'month_start'])
cluster_months = cluster_months.fillna(0)

cluster_months['weekly_hrs'] = cluster_months['hr_played']/4.3

_clustorder = cluster_months.groupby('cluster_name')['hr_played'].sum().sort_values(ascending=False).index
cluster_months['cluster_name'] = pd.Categorical(cluster_months['cluster_name'], categories=_clustorder, ordered=True)

cluster_months = cluster_months.sort_values(['cluster_name', 'month_start'])
cluster_months['6mo_avg'] = (
    cluster_months.groupby('cluster_name', observed=False)['weekly_hrs'].transform(lambda x: x.rolling(window=6).mean())
)

cluster_months['6mo_pct'] = cluster_months['6mo_avg'] / cluster_months['month_start'].map(cluster_months.groupby('month_start')['6mo_avg'].sum())

px_clustertrend=px.area(
    cluster_months,
    x='month_start',
    y='6mo_pct',
    color='cluster_name',
    line_shape = 'spline',
    color_discrete_sequence=_colors,
    labels = {
        'month_start': 'Month',
        '6mo_pct': 'Percent Listening',
        'cluster_name': 'Style'
    },
    hover_data={'6mo_pct':':.0%'},  #set formatting for hover
    title="All-Time Listening by Style"
)

px_clustertrend.update_traces(line_width=0.5)
px_clustertrend.update_layout(height=500)
px_clustertrend.update_yaxes(tickformat='.0%', range = [0,1])
px_clustertrend.update_layout(plot_bgcolor='white', xaxis_title='')

for f in [px_clustertrend.update_xaxes, px_clustertrend.update_yaxes]:  #iteratiely update both axis
    f(gridcolor='gainsboro', griddash='dot', gridwidth=0)


# In[177]:


#patterns in predominant style by time of day?
streamhx_topc = streamhx[streamhx['cluster_name'].isin(graph_clusters)]
_tots = streamhx_topc.groupby('cluster_name')['hr_played'].sum()
baseline = _tots / _tots.sum()

#for each cluster, what's its pct by weekday / hour?
hrly_tots = {}
for c in graph_clusters:
    _df = streamhx[streamhx['cluster_name']==c]
    _hmap = _df.pivot_table(index='hour', columns='weekday', values='hr_played', aggfunc='sum')
    _hmap = _hmap.reindex(  #ensure all have all col/rows and in same order
        index=range(0,24), 
        columns=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    )
    _hmap = _hmap.fillna(0)
    hrly_tots[c] = _hmap

hrly_denom = streamhx_topc.pivot_table(index='hour', columns='weekday', values='hr_played', aggfunc='sum')
hrly_denom = hrly_denom.reindex(
    index=range(0,24), 
    columns=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
)

hrly_styles = {}
for c, hmap in hrly_tots.items():   
    _pct = hmap / hrly_denom
    _diff = _pct / baseline[c] - 1
    
    _expected = hrly_denom * baseline[c]
    _zscore = (hmap - _expected) / np.sqrt(_expected)   # (observed-expected)/sqrt(expected)  (raw cts, not %)
    
    hrly_styles[c] = {
        'overall_pct': _pct,
        'pct_diff': _diff,
        'rolling_pctdiff': _diff.T.rolling(window=4, min_periods=1).mean().T,  #axis=1 deprecated here,
        'zscore': _zscore,
        'rolling_zscore': _zscore.T.rolling(window=3, min_periods=1).mean().T
    }


# In[178]:
from matplotlib.colors import to_hex, to_rgba
_colorsalpha = [to_rgba(c, alpha=0.5) for c in _colors]  #make them a little more transparent to match plotly


# each hour, which cluster is most above norm?
_targ = 'overall_pct'

stack = np.stack([hrly_styles[c][_targ] for c in hrly_styles])
maxid = np.argmax(stack, axis=0)  #idxmax is pd
topcat = np.array(list(hrly_styles))[maxid]

hmap = pd.DataFrame(maxid, columns=hrly_denom.columns)
hmap.index = hmap.index.map(timelabels)

f_style_overall_hmap , ax = plt.subplots(figsize=(9,7))
sns.heatmap(hmap, cmap=_colorsalpha, cbar=False, annot=topcat, fmt='', annot_kws={'size':5}, ax=ax, vmin=0, vmax=len(graph_clusters)-1)
ax.add_patch(wkday_patch())
ax.add_patch(wkend_patch())
ax.add_patch(border_patch())
ax.set_xlabel('')
ax.set_ylabel('')
ax.set_title("Hourly Most-Listened Styles")



# In[179]:


# each hour, which cluster is most above norm?
_targ = 'rolling_zscore'

stack = np.stack([hrly_styles[c][_targ] for c in hrly_styles])
maxid = np.argmax(stack, axis=0)  #idxmax is pd
topcat = np.array(list(hrly_styles))[maxid]

hmap = pd.DataFrame(maxid, columns=hrly_denom.columns)
hmap.index = hmap.index.map(timelabels)

f_style_diff_hmap, ax = plt.subplots(figsize=(9,7))
sns.heatmap(hmap, cmap=_colorsalpha, cbar=False, annot=topcat, fmt='', annot_kws={'size':5}, ax=ax, vmin=0, vmax=len(graph_clusters)-1)
ax.add_patch(wkday_patch())
ax.add_patch(wkend_patch())
ax.add_patch(border_patch())
ax.set_xlabel('')
ax.set_ylabel('')
ax.set_title("Hourly Highest Outlier Styles")


# In[180]:


from matplotlib.colors import LinearSegmentedColormap

_targ = 'rolling_zscore'

#find the most interesting style pattern
_ztot = [hrly_styles[s][_targ].abs().sum().sum() for s in hrly_styles]
_zrank = np.flip(np.argsort(_ztot))  #no reverse/ascending param in argsort



st_progress_text.text("✅ Done!")
st_progress_bar.progress(100)

time.sleep(1)  #let user see 100% for a sec
st_progress_text.empty()
st_progress_bar.empty()



#------------------------------------------------------------------------------------------------------------------------------
#hold all charts to the end and display here-----------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------
st.text("")

##Overview##
st.header("Overview")
st.write(f"Over your **{int(datadays/365)+1} years** on Spotify, you've listened to **{len(streamhx):,}** songs.  "  
         f"That's over **{int(streamhx['hr_played'].sum()):,} hours**... or *{streamhx['hr_played'].sum()/24:.0f} days* straight!"
         )

##Overall Listening Volum Trends##
st.write(f"Your peak all-time listening year was {best['startdt']:%b %d, %Y} - {best['enddt']:%b %d, %Y}, when you listened to an average of **{best['hours']/52:.1f} hrs/wk**.  "  
         f"Last year, you listened an average of **{lastyr/52:.1f} hrs/wk**. This is down *{-pct_change:.0%}* from your peak.  " 
         "Check out your full music listening history below:"
         )
st.plotly_chart(px_totalhrstrend)

##Top Artists##
st.text("")
st.header("Top Artists")
st.write(f"You've listened to a lot of different artists over the years ({streamhx['artist_name'].nunique():,} to be exact!)... but these were your favourites overall:")
st.dataframe(df_artists_display)

st.text("")
st.write("Check out the historic relationship you've had with each of these artists:")
st.plotly_chart(px_artisttrend)

##Top Songs##
st.header("Top Tracks")
st.write(f"Out of the {streamhx['song_name'].nunique():,} unique songs you've listened to, these ones stood out as your top tracks of all time:")
st.dataframe(df_topsongs_display)

st.text("")
st.write(f"There were also a few times you got a bit... *too* into one specific song for a week or so...")
st.dataframe(df_obsessions_display, height = 500)
st.write("...whoops  \n")

##Genre Clustering##
st.text("")
st.header("Musical Style")
st.write("For all that musical variety, the vast majority of your listening can be described as the following:")
st.plotly_chart(px_clusterpie)

st.write("Featuring favourite artists such as...")
_cols = st.columns(2)
for _i, c in enumerate(graph_clusters):
    _clustersample = tagged_artists[tagged_artists['cluster_name']==c].sort_values(by='hr_played', ascending=False).head(10)
    _df_display = _clustersample[['artist_name', 'hr_played']].reset_index(drop=True).rename(columns={
        'artist_name': "Artist",
        'hr_played': "Total Hours"
    })
    _df_display.index += 1
    _df_display = _df_display.style.format({
        "Total Hours": '{:.1f}'
    })
    with _cols[_i%2]: 
        st.subheader(c)
        st.dataframe(_df_display, height=200)  #alternate placing these in col1 and col2

st.text("")
st.write("But it goes without saying, your taste has evolved over time. Check out your historic listening patterns with each of your favourite styles:")
st.plotly_chart(px_clustertrend)

st.header("Listening Trends")

st.write(f"Whether it's early in the morning or late at night, everyone has a favourite time to listen to music.  Here are your top listening times:")
st.pyplot(f_hrsheatmap)

st.text("")
st.write("And at any given point in the day, this is the kind of music you're typically listening to...")
st.pyplot(f_style_overall_hmap)

#st.write("...But these are the genres you disproportionately gravitate towards at each point in the day:")
#st.pyplot(f_style_diff_hmap)

st.text("")
st.write("...But like all things, there's a time and a place... Here are some standout listening trends you have for specific styles throughout the week:")
for s in _zrank[:5]:  #print the top 5 trends
    _style = list(hrly_styles)[s]
    
    cmap = LinearSegmentedColormap.from_list("name", [(0,'red'), (0.4,'whitesmoke'), (0.6,'whitesmoke'), (1,'green')])
    hmap = hrly_styles[_style][_targ].copy()
    hmap.index = hmap.index.map(timelabels)
    
    fig, ax = plt.subplots()
    ax = sns.heatmap(hmap, cmap=cmap, center=0, vmin=-2, vmax=2, ax=ax)
    ax.add_patch(wkday_patch())
    ax.add_patch(wkend_patch())
    ax.add_patch(border_patch())
    ax.set_xlabel('')
    ax.set_ylabel('')

    cbar = ax.collections[0].colorbar  #override with generic labels, don't get into explaining zscores
    cbar.set_ticks([-2, 0, 2])
    cbar.set_ticklabels(["Less Than Usual", "Typical", "More Than Usual"])

    st.subheader(f"{_style} Listening Trends:")
    st.pyplot(fig)


for _ in range (4): st.text("")
st.subheader("That's all for now!")
st.write(f"For one last fun fact, in the time you've spent listening to Spotify, you could have coded this project {streamhx['hr_played'].sum()/40:.0f} times!!  "
    "Thanks so much for checking this out - it was a blast to build and I'd love to hear your thoughts or suggestions for future work.  "
    "  \n ~Hayden"
    )
