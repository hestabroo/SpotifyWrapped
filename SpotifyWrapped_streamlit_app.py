#!/usr/bin/env python
# coding: utf-8

# In[154]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[155]:


#could set this up later to work right from the zip with import zipfile
import zipfile, glob, json

zippath = "/Users/haydenestabrook/Documents/Data Science Personal Portfolio/20250811 Spotify Wrapped/20250819_ExntendedStreamHistory.zip"

with zipfile.ZipFile(zippath, "r") as z:
    files = [f for f in z.namelist() if f.startswith("Spotify Extended Streaming History/Streaming_History_Audio") and f.endswith(".json")]
    files = sorted(files)

    dfs=[]
    for f in files:
        with z.open(f) as fo:
            data = pd.DataFrame(json.load(fo))
            dfs.append(data)

streamhx = pd.concat(dfs).reset_index()


# In[156]:


#helpful stuff
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


# # Top Artists

# In[157]:


artist_month[artist_month['artist_name']=="Marianas Trench"]


# In[158]:


#basic stuff, top artists
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
artists


# In[159]:


#try plotly for interactive
import plotly.express as px

artist_month['hrs_perweek'] = artist_month['hr_played'] / 4.3  #avg weeks/month

artist_month['artist_name'] = pd.Categorical(artist_month['artist_name'], categories=artists.index, ordered=True)  #convert to categorical to allow sorting by top artists
artist_month = artist_month.sort_values(['artist_name', 'month_start'])  #sort (required upfront for accurate rolling)

artist_month['6mo_avg'] = (  #calculate a 6mo avg
    artist_month.groupby('artist_name', observed=False)['hrs_perweek'].transform(lambda x: x.rolling(window=6).mean())
)

fig = px.area(
    artist_month,
    x='month_start',
    y='6mo_avg',
    color='artist_name',
    line_shape='spline',
    labels = {
        'month_start': 'Month',
        'artist_name': 'Artist',
        '6mo_avg': 'Avg. Weekly Hours'
    },
    hover_data={'6mo_avg':':.1f'}  #set formatting for hover
)

fig.update_traces(line_width=0.5)
fig.update_layout(plot_bgcolor='white', xaxis_title='')

for f in [fig.update_xaxes, fig.update_yaxes]:  #iteratiely update both axis
    f(gridcolor='gainsboro', griddash='dot', gridwidth=0)
    
fig.show()


# # Top Songs

# In[160]:


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
songs.head(20)


# In[161]:


#what were your obsessions??
threshold = np.percentile(songweeks.groupby(by='week_start')['times_played'].max(), 95)  #make this dynamic to different listening styles
obsessions = songweeks[(songweeks['times_played']>=threshold) & (songweeks['total_hrs']>threshold*1/60)]  #there seem to be weird weeks with lots of very short plays... exclude

#just take the top 1 song per week
_topids = obsessions.groupby('week_start')['times_played'].idxmax()
obsessions = obsessions.loc[_topids]
obsessions.sort_values(by='week_start', inplace=True)
obsessions


# In[ ]:





# # Listening Times & Patterns

# In[162]:


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

print(f"\nYour peak listening year was {best['startdt']:%b %d, %Y} - {best['enddt']:%b %d, %Y}, when you listened to an average of {best['hours']/52:.1f} hrs/wk.\n")

#last year
lastyr = weekly.iloc[-52:]['hr_played'].sum()
pct_change = (lastyr - best['hours']) / best['hours']

print(f"\nLast year, you listened an average of {lastyr/52:.1f} hrs/wk.  This is down {-pct_change:.0%} from your peak.\n")


# In[163]:


#maybe plotly
import plotly.express as px

_peakyn = [True if best['start'] <= x <= best['end'] else False for x in weekly.index]
colors = ['Peak' if x else 'Weekly Hours' for x in _peakyn]

fig = px.bar(
    weekly,
    x='week_start',
    y='hr_played',
    color=colors
)

fig.add_scatter(
    x=weekly['week_start'],
    y=weekly['6mo_avg'],
    mode='lines',
    line={'color':'black', 'dash':'dot', 'width':1.5},
    name='6mo Avg.'
)

fig.show()


# In[ ]:





# In[164]:


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

plt.figure(figsize=(8,6))
ax = sns.heatmap(hmap, annot=False, fmt='.0%', cmap = 'coolwarm', annot_kws={'size':8})

#define these patches so we can reuse them later
def wkday_patch():
    return patches.Rectangle(  #workday
        xy=(0,17),
        width=7,
        height=-8,
        fill=False,
        edgecolor='black',
        lw=1
    )

def wkend_patch():
    return patches.Rectangle(  #weekend
        xy=(5,24),
        width=2,
        height=-24,
        fill=False,
        edgecolor='black',
        lw=1
    )

def border_patch():
    return patches.Rectangle(  #full border
        xy=(0,24),
        width=7,
        height=-24,
        fill=False,
        edgecolor='black',
        lw=1
    )

ax.add_patch(wkday_patch())
ax.add_patch(wkend_patch())
ax.add_patch(border_patch())


# In[165]:


import calendar

timedata['month_fmt'] = timedata['month_start'].dt.strftime('%B')

hmap=timedata.pivot_table(index='year', columns='month_fmt', values='hr_played', aggfunc = lambda x: x.sum()/4.3)  #weekly hours
hmap = hmap[calendar.month_name[1:]].sort_index()  #sort months & years

plt.figure(figsize=(12,4))
sns.heatmap(hmap, annot=True, fmt='.1f', cmap='coolwarm', annot_kws={'size':8})


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

px.pie(
    platform,
    names = 'platform_cat',
    values='hr_played'
)


# In[167]:


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

fig=px.area(
    platform_full,
    x='month_start',
    y='6mo_avg',
    color='platform_cat',
    line_shape = 'spline'
)

fig.update_traces(line_width=0.5)

fig.show()


# # Metadta and Sentiment Analysis

# In[168]:


#DAMM - spotify's audio features API has been disabled for free users... pivot to last.fm
#last fm data is much more populated by ARTIST, not track
#**limit to the artists making up x% of playtime?  better trends and save the API
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

responses = {}
_c = 0  #tracking
for a in topartists:
    params['artist'] = a
    _resp = requests.get(url, params)
    responses[a] = _resp.json()

    if _c%50==0: print(f"{_c}/{len(topartists)}")
    _c+=1


# In[170]:


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
artist_tags_long = artist_tags_long[~artist_tags_long['tag_name'].isin(['All', 'Canadian'])]  #get rid of that garbage "All" tag

#cut out noisy tags
tags_all = artist_tags_long['tag_name'].value_counts()  #cut down noisy tags
tags_keep = tags_all.sort_values(ascending=False).head(100).index  #top 100 only
artist_tags_long = artist_tags_long[artist_tags_long['tag_name'].isin(tags_keep)] 

artist_tags = artist_tags_long.pivot_table(index='artist_name', columns='tag_name', values='ct', aggfunc='max')  #convert to wide format  #for some reason, some artists have the same tag twice... use aggfunc=max
artist_tags = artist_tags.fillna(0)
artist_tags.shape


# In[ ]:





# ## Clustering (KMeans, (H)DBScan)

# In[171]:


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
_nrange = range(10,31)  #max 10-30 clusters
for n in _nrange:
    kmodel = Pipeline([
        ('scaler', StandardScaler()),
        #('pca', PCA(n_components=int(artist_tags.shape[1]/3))),
        ('kmeans', KMeans(n_clusters=n))
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

print(f"\nn={n}\n\nInertia: {results[n]['inertia']:.3f}\nDBI: {results[n]['DBI']:.3f}\n")

pd.DataFrame.from_dict(results, orient='index').plot(y=['inertia', 'silhouette', 'DBI'], secondary_y=['inertia'])


# In[172]:


kmodel = results[n]['model']  #retrieve the optimum model

clusters = pd.DataFrame(
    kmodel.named_steps['kmeans'].cluster_centers_,
    columns = artist_tags.columns
)

plt.figure(figsize=(14,10))
sns.heatmap(clusters.T, annot=True, fmt='.2f', annot_kws={'size':5})


# In[ ]:





# In[ ]:





# In[173]:


#add names back into the artists
naming_threshold = np.mean(clusters.max())/10  #could be different for different data
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

for c, _ in tagged_artists.groupby('cluster_name')['hr_played'].sum().sort_values(ascending=False).items():
    display(tagged_artists[tagged_artists['cluster_name']==c].sort_values(by='hr_played', ascending=False).head(10))


# In[174]:


#put labels back into streamhx
artist_labels = tagged_artists.set_index('artist_name')['cluster_name']
streamhx['cluster_name'] = streamhx['artist_name'].map(artist_labels)

_crank = streamhx.groupby('cluster_name')['hr_played'].sum().sort_values(ascending=False)
_cumpct = _crank.cumsum() / _crank.sum()

graph_clusters = _cumpct[_cumpct<=0.90].index  #only include categories that explain 90% of listening
graph_clusters = [c for c in graph_clusters if c != 'Other']  #don't bother showing that "other" either if it shows up


# In[175]:


#add a straight up pie chart
_graph = streamhx[streamhx['cluster_name'].isin(graph_clusters)]

fig = px.pie(
    names=_graph['cluster_name'],
    values=_graph['hr_played'],
    color_discrete_sequence=_colors    
)

fig.show()


# In[176]:


_grp = streamhx.groupby(by=['month_start', 'cluster_name'], as_index=False)['hr_played'].sum()

#for smooth 6mo avgs, will need to fill in missing months per cluster
months = streamhx['month_start'].unique()
#just top 10?
_climit = 100
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

_colors = sns.color_palette('Paired', len(clusters)).as_hex()

fig=px.area(
    cluster_months,
    x='month_start',
    y='6mo_pct',
    color='cluster_name',
    line_shape = 'spline',
    color_discrete_sequence=_colors
)

fig.update_traces(line_width=0.5)
fig.update_layout(height=500)
fig.show()


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


# each hour, which cluster is most above norm?
_targ = 'overall_pct'

stack = np.stack([hrly_styles[c][_targ] for c in hrly_styles])
maxid = np.argmax(stack, axis=0)  #idxmax is pd
topcat = np.array(list(hrly_styles))[maxid]

hmap = pd.DataFrame(maxid, columns=hrly_denom.columns)
hmap.index = hmap.index.map(timelabels)

print("\nHere's what you're normally listening to:")

plt.figure(figsize=(9,7))
ax = sns.heatmap(hmap, cmap=_colors, cbar=False, annot=topcat, fmt='', annot_kws={'size':5})
ax.add_patch(wkday_patch())
ax.add_patch(wkend_patch())
ax.add_patch(border_patch())


# In[179]:


# each hour, which cluster is most above norm?
_targ = 'rolling_zscore'

stack = np.stack([hrly_styles[c][_targ] for c in hrly_styles])
maxid = np.argmax(stack, axis=0)  #idxmax is pd
topcat = np.array(list(hrly_styles))[maxid]

hmap = pd.DataFrame(maxid, columns=hrly_denom.columns)
hmap.index = hmap.index.map(timelabels)

print("\nEach hour, these are the styles you most disproportionately listen to:")

plt.figure(figsize=(9,7))
ax = sns.heatmap(hmap, cmap=_colors, cbar=False, annot=topcat, fmt='', annot_kws={'size':5})  #base
ax.add_patch(wkday_patch())
ax.add_patch(wkend_patch())
ax.add_patch(border_patch())


# In[180]:


from matplotlib.colors import LinearSegmentedColormap

_targ = 'rolling_zscore'

#find the most interesting style pattern
_ztot = [hrly_styles[s][_targ].abs().sum().sum() for s in hrly_styles]
_zrank = np.flip(np.argsort(_ztot))  #no reverse/ascending param in argsort

for s in _zrank[:5]:  #print the top 5 trends
    _style = list(hrly_styles)[s]
    
    cmap = LinearSegmentedColormap.from_list("name", [(0,'red'), (0.4,'whitesmoke'), (0.6,'whitesmoke'), (1,'green')])
    hmap = hrly_styles[_style][_targ].copy()
    hmap.index = hmap.index.map(timelabels)
    
    print(f"{_style} listening patterns:\n")
    
    ax = sns.heatmap(hmap, cmap=cmap, center=0, vmin=-2, vmax=2)
    ax.add_patch(wkday_patch())
    ax.add_patch(wkend_patch())
    ax.add_patch(border_patch())
    plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




