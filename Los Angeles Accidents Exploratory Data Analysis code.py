#!/usr/bin/env python
# coding: utf-8

# # Exporatory Data Analysis: LA Car Accidents (2017-2022)
# This analysis delves into the temporal, spatial, and environmental factors of car accidents in Los Angeles from 2017-2022, aiming to uncover valuable insights for future road safety initiatives. 
# 
# ### **Key areas of investigation:**
# 
# - **Temporal patterns:** Examining seasonality, day-of-week variations, and time-of-day trends in accident occurrence
# - **Spatial distribution:** Utilizing interactive maps to visualize accident locations by year and by traffic severity level
# - **Environmental factors:** Exploring the potential link between weather conditions and accident occurrence
# 
# Further research will utilize statistical hypothesis testing and comparative city assessments to enhance these preliminary insights and provide a more detailed understanding to guide future initiatives.
# 

# ## 📝 Data preparation

# In[1]:


import numpy as np
import pandas as pd

import seaborn as sns
sns.set_style("darkgrid")
import matplotlib.pyplot as plt

import folium
from folium import plugins
from folium.plugins import HeatMap
from folium.plugins import MarkerCluster    
from folium import LayerControl


# In[2]:


#download data
get_ipython().system('pip install opendatasets --upgrade --quiet')


# In[3]:


import opendatasets as od 

download_url = 'https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents'
od.download(download_url)


# In[4]:


data_filename = './us-accidents/US_Accidents_March23.csv'


# In[5]:


df = pd.read_csv(data_filename)


# In[6]:


df


# In[7]:


df_backup = df.copy()


# In[8]:


len(df.columns)


# In[9]:


len(df)


# In[10]:


df.info()


# In[11]:


#change Start_Time from object to DateTime datatype
df['Start_Time'] = pd.to_datetime(df['Start_Time'])


# In[12]:


df.describe()


# In[13]:


#find null value percentage for each column
missing_percentages = df.isna().sum().sort_values(ascending=False) / len(df) *100
missing_percentages


# ## 📝 Cities Exploration

# In[14]:


cities = df.City.unique()
len(cities)


# In[15]:


# Calculate the number of accidents for each city
accidents_by_city = df['City'].value_counts()
accidents_by_city


# In[16]:


# Calculate the percentage of accidents for each city in relation to the total number of accidents
total_accidents = len(df)
accidents_percentage_by_city = accidents_by_city / total_accidents * 100
accidents_percentage_by_city


# In[17]:


# display top 10 cities with the highest number of accidents
display(accidents_by_city[:10])

# plot top 10 cities with highest accident count
accidents_by_city[:20].plot(kind='barh')


# ## 📝 Create Los Angeles Accident DataFrame

# In[18]:


df_la = df[df.City.str.lower().isin(['los angeles', 'la', 'losangeles'])]


# In[19]:


len(df_la)


# In[20]:


df_la


# ## 📝 Calculate accidents per capita for each year

# In[21]:


# Calculate LA car accident count per year
accidents_by_year_la = df_la.Start_Time.dt.year.value_counts()
accidents_by_year_la.sort_values(ascending = False)


# In[22]:


# visualize LA car accidents per year
sns.displot(df_la.Start_Time.dt.year, bins = 8)


# #### Note:
# Omit 2016 and 2023 data as these years are incomplete.
# 
# According to data source (https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents), 2016 data collection started in February. The last recorded accident in LA was March 31, 2023.

# In[23]:


# create a df that omits data from 2016 and 2023 and contains only accidents from LA
df_la = df[
        (df.City.str.lower().isin(['los angeles', 'la', 'losangeles']))
        & (~df.Start_Time.dt.year.isin([2016, 2023]))
        ]


# In[24]:


# verify df_la is correct by verifing the following:
# min and max start dates in dataframe are within years 2017-2022
# only accidents in Los Angeles city are included in data frame
print(f"Min start date: {min(df_la.Start_Time)}")
print(f"Max start date: {max(df_la.Start_Time)}")
print(f"Cities included in df_la: {df_la.City.unique()}")


# In[25]:


# visualize accident_count by year
sns.displot(df_la.Start_Time.dt.year, bins = 6)


# In[26]:


#Prepare to calculate accidents per capita
#LA_population source: https://www.census.gov/quickfacts/fact/table/losangelescitycalifornia/PST045222
census_data = {
    'Year': [
        2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022
    ], 
    'LA_population': [
        3792621, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 
                np.nan, np.nan, np.nan, 3898747, np.nan, 3822238
    ]
}
census_data_df = pd.DataFrame(census_data)
census_data_df


# In[ ]:





# In[27]:


# Utilizing available census data, piecewise linear interpolation provides an estimate 
# of LA's population in data gaps.
la_population_inter_df = census_data_df.interpolate()
la_population_inter_df


# In[28]:


# Plot the estimated Los Angeles population over time
x_data = la_population_inter_df.Year
y_data = la_population_inter_df.LA_population
plt.plot(x_data, y_data)
plt.xlabel('Year')
plt.ylabel('LA Population')
plt.title('Estimated Los Angeles Population Over Time')
plt.ylim(3_700_000, 4_000_000)
plt.show()


# In[29]:


# display a table containing Year, LA_population, Accident_Count, and Accidents_Per_Capita

# Calculate the number of accidents per year
accidents_by_year_la = df_la.Start_Time.dt.year.value_counts()

# Convert the Series to a DataFrame
accidents_by_year_la_df = accidents_by_year_la.reset_index()
accidents_by_year_la_df.columns = ['Year', 'Accident_Count']

# Display table with Year and Accident_Count
display(accidents_by_year_la_df)


# In[74]:


# Merge accident counts with population data
merged_la_accidents_population = pd.merge(la_population_inter_df, accidents_by_year_la_df, on='Year')

# Calculate accidents per capita and add a new column
merged_la_accidents_population['accidents_per_capita'] = (
    merged_la_accidents_population['Accident_Count'] / merged_la_accidents_population['LA_population'] * 100000
)
display(merged_la_accidents_population)


# In[31]:


# visualize accidents per capita 
merged_la_accidents_population.plot.bar(x='Year', y='accidents_per_capita', rot=0)


# ## 🚦 Insight:
# * 2022 was the year with highest number of accidents per capita.
# * 2017 was the year with lowest number of accidents per capita.

# ## 📍 Create an Interactive Map to Evaluate LA Traffic Accidents (2017-2022) 
# **Map User Instructions:** 
# * Use the Layer tool on map to filter accidents by year
# * Zoom in on the map to observe accident points at specific intersections and roads of interest

# In[73]:


# create a Folium map centered on the first accident in the Los Angeles dataset, 
# add zoom level of 10 and OpenStreetMap tiles.
m = folium.Map(
    location=[df_la['Start_Lat'].iloc[0], df_la['Start_Lng'].iloc[0]], 
    zoom_start=10, control_scale=True, tiles='OpenStreetMap'
)

# create separate MarkerCluster objects for each year of accidents (2017-2022) and add them to the map.
df_la_2017 = df_la[df_la['Start_Time'].dt.year == 2017]
df_la_2017_marker_cluster = plugins.MarkerCluster(name='2017 Accidents').add_to(m)

# Loop through each accident in 2017 and add a marker with lat/lng coordinates and popup information.
for idx, row in df_la_2017.iterrows():
    folium.Marker(location=[row['Start_Lat'], row['Start_Lng']], 
    popup=f"Latitude: {row['Start_Lat']}, Longitude: {row['Start_Lng']}").add_to(df_la_2017_marker_cluster)

    
df_la_2018 = df_la[df_la['Start_Time'].dt.year == 2018]
df_la_2018_marker_cluster = plugins.MarkerCluster(name='2018 Accidents').add_to(m)

for idx, rows in df_la_2018.iterrows():
    folium.Marker(location=[row['Start_Lat'], row['Start_Lng']], 
    popup=f"Latitude: {row['Start_Lat']}, Longitude: {row['Start_Lng']}").add_to(df_la_2018_marker_cluster)


df_la_2019 = df_la[df_la['Start_Time'].dt.year == 2019]
df_la_2019_marker_cluster = plugins.MarkerCluster(name='2019 Accidents').add_to(m)

for idx, rows in df_la_2019.iterrows():
    folium.Marker(location=[row['Start_Lat'], row['Start_Lng']], 
    popup=f"Latitude: {row['Start_Lat']}, Longitude: {row['Start_Lng']}").add_to(df_la_2019_marker_cluster)

df_la_2020 = df_la[df_la['Start_Time'].dt.year == 2020]
df_la_2020_marker_cluster = plugins.MarkerCluster(name='2020 Accidents').add_to(m)

for idx, rows in df_la_2020.iterrows():
    folium.Marker(location=[row['Start_Lat'], row['Start_Lng']], 
    popup=f"Latitude: {row['Start_Lat']}, Longitude: {row['Start_Lng']}").add_to(df_la_2020_marker_cluster)


df_la_2021 = df_la[df_la['Start_Time'].dt.year == 2021]
df_la_2021_marker_cluster = plugins.MarkerCluster(name='2021 Accidents').add_to(m)


for idx, rows in df_la_2021.iterrows():
    folium.Marker(location=[row['Start_Lat'], row['Start_Lng']], 
    popup=f"Latitude: {row['Start_Lat']}, Longitude: {row['Start_Lng']}").add_to(df_la_2021_marker_cluster)

df_la_2022 = df_la[df_la['Start_Time'].dt.year == 2022]
df_la_2022_marker_cluster = plugins.MarkerCluster(name='2022 Accidents').add_to(m)

for idx, rows in df_la_2022.iterrows():
    folium.Marker(location=[row['Start_Lat'], row['Start_Lng']], 
    popup=f"Latitude: {row['Start_Lat']}, Longitude {row['Start_Lng']}").add_to(df_la_2022_marker_cluster)

# Create a LayerControl object to allow users to switch between accident year layers.
layer_control = LayerControl(position='topright')
layer_control.add_to(m)

# Add a custom JavaScript snippet to initially highlight the 2017 accident layer.
# This changes the text of the layer toggle button to "2017 Accidents".
filter_js = """
    <script>
        document.getElementsByClassName('leaflet-control-layers-toggle')[0].innerHTML = '2017 Accidents';
    </script>
"""

folium.Element(filter_js).add_to(m)

#m


# In[67]:


# Interactive Map Demonstration (Screenshots Due to Map Size)
from IPython.display import display, Image

map = "/Users/austinshirk/Desktop/la_map.png"

display(Image(filename=map))

# filter by year of interest at top right of map view before zooming in to assess individual accident points


# In[51]:


map2 = "/Users/austinshirk/Desktop/la_map2.png"

display(Image(filename=map2))

# zoom into particular road of interest


# In[68]:


map3 = "/Users/austinshirk/Desktop/la_map3.png"
display(Image(filename=map3))

# Select a circular icon (e.g. 71) to expand and view all nearby accident instances


# In[54]:


map4 = "/Users/austinshirk/Desktop/la_map4.png"

display(Image(filename=map4))
# select a pin to view the accident's latitude and longitude


# ## 📝 Compare LA car accidents over time

# **Calculate the following:**
# 1. month with highest/lowest accident count
# 2. day of week with highest/lowest accident count
# 3. time of day with highest/lowest accident count

# In[33]:


# Compute the number of accidents across the 12 months in LA (January=1, December=12)
la_accidents_month = df_la.Start_Time.dt.month
la_monthly_counts = la_accidents_month.value_counts().sort_index()

# Compute the number of accidents across days of the week in LA (Monday=0, Sunday=6)
la_accidents_week = df_la.Start_Time.dt.day_of_week
la_weekly_counts = la_accidents_week.value_counts().sort_index()

# Compute the number of accidents across hours of the day in LA (12 AM=0, 11 PM=23)
la_accidents_hour = df_la.Start_Time.dt.hour
la_hourly_counts = la_accidents_hour.value_counts().sort_index()


# In[34]:


# Create comparative bar charts exploring monthly, weekly, and hourly patterns of car accidents in LA.
fig, axes = plt.subplots(1, 3, figsize=(18, 7))

# Plotting for subplot 1
axes[0].bar(la_monthly_counts.index, la_monthly_counts, color='skyblue')
axes[0].set_title('Accidents By Month', fontsize=16)
axes[0].set_xlabel('Month', fontsize=13)
axes[0].set_ylabel('Number of Accidents', fontsize=14)

# Plotting for subplot 2
axes[1].bar(la_weekly_counts.index, la_weekly_counts, color='lightcoral')
axes[1].set_title('Accidents By Day of Week', fontsize=16)
axes[1].set_xlabel('Day', fontsize=13)
#axes[1].set_ylabel('Number of Accidents')

# Plotting for subplot 3
axes[2].bar(la_hourly_counts.index, la_hourly_counts, color='lightgreen')
axes[2].set_title('Accidents By Hour', fontsize=16)
axes[2].set_xlabel('Hour', fontsize=13)
#axes[2].set_ylabel('Number of Accidents')

plt.tight_layout()

plt.show()


# ## 🚦 Insight
# ####  Accidents By Month
# - The month with the highest number of accidents is December
# - The month with the lowest number of accidents is July.
# #### Accidents By Day of Week
# - The highest number of car accidents occured on Friday
# - The least number of car accidents occured on Sunday
# #### Accidents By Hour
# - The top five hourly accident counts occurred between 1 PM and 6 PM
# - The lowest five hourly accident counts occured between 12 AM to 5 AM

# ## 📝 Evaluate Traffic Flow Disruption Caused by Car Accidents
# 

# **Note:** 
# 
# The Severity Column in the DataFrame shows the severity of the accident, a number between 1 and 4, where 1 indicates the least impact on traffic (i.e., short delay as a result of the accident) and 4 indicates a significant impact on traffic (i.e., long delay).

# In[35]:


# creates a crosstab to analyze traffic severity by hour of the day.
severity_counts = pd.crosstab(df_la.Start_Time.dt.hour, df_la.Severity)

# sum the counts across severity levels and reset the index
severity_counts_df = pd.DataFrame(severity_counts.sum().reset_index())
severity_counts_df.columns = ['Traffic_Severity', 'Accident_Count']
display(severity_counts_df)


fig2, ax2 = plt.subplots(4, 1, figsize=(15, 25), gridspec_kw={'hspace': 0.4})
red_palette = sns.color_palette("Reds", n_colors=4)

# loops through each severity level (1-4):
for i in range(4):
    bars = ax2[i].bar(severity_counts.index, severity_counts[i + 1], color=red_palette[i], label=f'Severity {i + 1}')
    ax2[i].set_xlabel('Hour of the Day', color='grey', fontsize=13)
    ax2[i].set_ylabel('Count of Accidents', color='grey', fontsize=13)
    ax2[i].set_title(f'Traffic Severity {i + 1} Accidents by Hour of the Day', fontsize=16)
    ax2[i].set_xticks(severity_counts.index)
    
plt.show()


# ## 🚦Insight
# - Traffic Severity 2 was the most common severity level in LA
# - The highest frequency of Traffic Severity 2 accidents occured between 1 PM and 6 PM 

# ## 📍 Create an Interactive Map to Evaluate LA Car Accidents by Traffic Severity
# **Map User Instructions:** 
# * Use the Layer tool on map to filter accidents by traffic severity level
# * Zoom in on the map to observe accident points at specific intersections and roads of interest

# In[71]:


# create a Folium map centered on the first accident in the Los Angeles dataset, 
# add zoom level of 10 and OpenStreetMap tiles.
m_1 = folium.Map(
    location=[df_la['Start_Lat'].iloc[0], df_la['Start_Lng'].iloc[0]],
    zoom_start=10, control_scale=True, tiles='OpenStreetMap'
)

# create separate MarkerCluster objects for each severity level and add them to the map.
sev_1_df = df_la[df_la.Severity == 1]
sev_1_marker_cluster = plugins.MarkerCluster(name='Severity_1').add_to(m_1)

# Loop through each accident with traffic severity level 1 and 
# add a marker with lat/lng and popup information.
for idx, row in sev_1_df.iterrows():
    folium.Marker(location=[row['Start_Lat'], row['Start_Lng']], 
    popup=f"Latitude: {row['Start_Lat']}, Longitude: {row['Start_Lng']}").add_to(sev_1_marker_cluster)

sev_2_df = df_la[df_la.Severity == 2]
sev_2_marker_cluster = plugins.MarkerCluster(name='Severity_2').add_to(m_1)

for idx, row in sev_2_df.iterrows():
    folium.Marker(location=[row['Start_Lat'], row['Start_Lng']], 
    popup=f"Latitude: {row['Start_Lat']}, Longitude: {row['Start_Lng']}").add_to(sev_2_marker_cluster)

    
sev_3_df = df_la[df_la.Severity == 3]
sev_3_marker_cluster = plugins.MarkerCluster(name='Severity_3').add_to(m_1)

for idx, row in sev_3_df.iterrows():
    folium.Marker(location=[row['Start_Lat'], row['Start_Lng']], 
    popup=f"Latitude: {row['Start_Lat']}, Longitude: {row['Start_Lng']}").add_to(sev_3_marker_cluster)

sev_4_df = df_la[df_la.Severity == 4]
sev_4_marker_cluster = plugins.MarkerCluster(name='Severity_4').add_to(m_1)

for idx, row in sev_4_df.iterrows():
    folium.Marker(location=[row['Start_Lat'], row['Start_Lng']], 
    popup=f"Latitude: {row['Start_Lat']}, Longitude: {row['Start_Lng']}").add_to(sev_4_marker_cluster)

# Create a LayerControl object to allow users to switch between traffic severity levels
layer_control = LayerControl(position='topright')
layer_control.add_to(m_1)

# Add a custom JavaScript snippet to initially highlight the 2017 accident layer.

# This changes the text of the layer toggle button to "Severity_1".

custom_js = """
    <script>
        document.getElementsByClassName('leaflet-control-layers-toggle')[0].innerHTML = 'Severity_1';
    </script>
"""

folium.Element(custom_js).add_to(m_1)

#m_1


# In[66]:


# Interactive Map Demonstration (Screenshots Due to Map Size)
map5 = "/Users/austinshirk/Desktop/la_map5.png"

display(Image(filename=map5))

# filter by traffic severity level at top right of map view before zooming in to assess individual accident points


# In[61]:


map2 = "/Users/austinshirk/Desktop/la_map2.png"

display(Image(filename=map2))

# zoom into particular road of interest


# In[69]:


map3 = "/Users/austinshirk/Desktop/la_map3.png"
display(Image(filename=map3))

# Select a circular icon (e.g. 71) to expand and view all nearby accident instances


# In[63]:


map4 = "/Users/austinshirk/Desktop/la_map4.png"

display(Image(filename=map4))
# select a pin to view the accident's latitude and longitude


# ## 📝 Evaluate accident counts by the top 10 weather conditions

# In[37]:


# view the top 10 weather conditions based on accident count
df_la.Weather_Condition.value_counts().head(10).reset_index()


# In[38]:


# create an la_weather_df 
la_weather_df = pd.DataFrame(
    df_la.Weather_Condition.value_counts().head(10)
).reset_index().rename(
    columns={
        'index':'Weather_Condition', 'Weather_Condition':'Count'
    }
)
la_weather_df


# In[39]:


# add percentage column to la_weather_df
top_weather_sum = la_weather_df['Count'].sum()

weather_percentage = []

for i in range(la_weather_df.shape[0]):
    pct = la_weather_df['Count'][i] / top_weather_sum * 100
    weather_percentage.append(round(pct, 2))

la_weather_df['Percentage(%)'] = weather_percentage

la_weather_df


# In[40]:


## Creates a bar chart for accident counts by weather condition
plt.figure(figsize=(10, 6))
colors = plt.cm.cividis(np.linspace(0.4, 1, len(la_weather_df)))
graph = plt.bar(la_weather_df.Weather_Condition, la_weather_df.Count, color=colors)

plt.xlabel('Weather Condition', fontsize = 15, color = 'grey')
plt.ylabel('Accident Count', fontsize = 15, color = 'grey')
plt.title('Accident Counts by Weather Condition in LA (2017-2022)', fontsize=18, color = 'black')

i=0

# Loops through each bar in the graph:
for p in graph:
    
    width = p.get_width()
    height = p.get_height()
    x, y = p.get_xy()
     
    plt.text(x+width/2,
             y+height*1.01,
             str(la_weather_df['Percentage(%)'][i])+'%',
             ha='center',
             weight='bold')
    
    i+=1

    plt.xticks(rotation=45, ha='right')

plt.show()


# ## 🚦Insight
# - In most accident cases (51.15%), the weather was Fair and in ~16% cases, the weather was Clear

# ## 📝 Evaluate accident counts by temperature (F)

# In[41]:


# Extract the minimum and maximum temperatures in the 'Temperature(F)' column
print(f"min: {min(df_la['Temperature(F)'])}")
print(f"max: {max(df_la['Temperature(F)'])}")


# In[42]:


# Define temperature bins and corresponding labels for categorization
bins = [30, 40, 50, 60, 70, 80, 90, 100, 110]
temp_groups = ['30-40', '40-50', '50-60', '60-70', '70-80', '80-90', '90-100', '100-110']


# Create a new column named 'Temperature_Category'
# Use the `pd.cut` function to categorize each temperature value in the 'Temperature(F)' column
# based on the defined bins and labels. The `right=False` argument ensures categories are exclusive.
df_la['Temperature_Category'] = pd.cut(df_la['Temperature(F)'], bins, labels = temp_groups, right=False)


# In[43]:


# Count the occurrences of each temperature category (`Temperature_Category`) using `value_counts()`
# Resets the index to create a new dataframe named `temperature_df`
temperature_df = df_la.Temperature_Category.value_counts().reset_index()
temperature_df.columns =['Temperature_Range(F)', 'Accident_Count']

# Sorts the dataframe by 'Temperature Range' in ascending order and resets the index again, dropping the old one.
temperature_df = temperature_df.sort_values(by='Temperature_Range(F)').reset_index(drop=True)
display(temperature_df)

# Creates a bar chart showing accident counts for each temperature range
plt.figure(figsize=(10, 6))
red_colors = plt.cm.Reds(np.linspace(0.1, 1, len(temperature_df)))
plt.bar(temperature_df['Temperature_Range(F)'], temperature_df['Accident_Count'], color=red_colors)
plt.xlabel('Temperature (F)', fontsize=15, color='grey')
plt.ylabel('Accident Count', fontsize=15, color='grey')
plt.title('Accident Counts By Temperature (F)', fontsize=18)
plt.show()


# ## 🚦Insight
# 
# - Accident frequency peaked within the 60-70°F range

# # 📊 Conclusion
# 
# This analysis offers insights into the temporal, spatial, and environmental factors of car accidents in Los Angeles from 2017-2022. These findings guide future actions in road safety and prompt further exploration through statistical hypothesis testing and comparative city assessments.
# 
# Key insights from this analysis include the following:
# 
# 
# **Temporal:**
# 
# * In December, the accident count reached its peak, while July had the lowest accident count.
# * Fridays had the highest accident count, while Sundays had the lowest.
# * The top five hourly accident counts occurred between 1 PM and 6 PM, while the lowest five were observed between 12 AM and 5 AM.
# 
# **Spatial Distribution:**
# 
# * Interactive maps reveal the accident coordinates filtered by year and traffic severity.
# 
# **Further Insights:**
# 
# * Fair weather conditions accounted for the majority of accidents.
# * Accidents per capita peaked in 2022.
# * 60-70°F temperature range had the highest accident count.
# 
# ## **Limitations:**
# 
# 
# It is important to assess whether there were significant changes in data collection techniques during the years of compiling the car accident data. This knowledge plays a crucial role in evaluating the reliability of the data. Additionally, it may be useful to examine data collection records to assess whether there were prolonged network connectivity issues that could have influenced the data quality.
# 
# ##### Note from dataset author:
# 
# "Please note that the dataset may be missing data for certain days, which could be due to network connectivity issues during data collection."
# 
# 
# 
# ## **Future Directions:**
#  
# * Conduct statistical hypothesis testing to further analyze the impact of temporal, environmental, and roadway factors on the accident rate.
# * Compare Los Angeles accident data with other cities to identify similarities and differences.
# * Identify zip codes exhibiting a high accident rate per capita through a comprehensive population analysis.

# # Data Acknowledgement:
# - Moosavi, Sobhan, Mohammad Hossein Samavatian, Srinivasan Parthasarathy, and Rajiv Ramnath. “A Countrywide Traffic Accident Dataset.”, 2019.
# 
# - Moosavi, Sobhan, Mohammad Hossein Samavatian, Srinivasan Parthasarathy, Radu Teodorescu, and Rajiv Ramnath. "Accident Risk Prediction based on Heterogeneous Sparse Data: New Dataset and Insights." In proceedings of the 27th ACM SIGSPATIAL International Conference on Advances in Geographic Information Systems, ACM, 2019.
