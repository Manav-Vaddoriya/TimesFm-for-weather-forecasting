## **Time Series Foundational Model**
There are many foundational models for textual data that handle long documents with long-range dependencies. Similarly, in time-series data, today’s forecast depends on past trends and historical patterns. Compared to traditional models such as LSTM and RNN, Transformers provide significantly better performance when modeling long-range dependencies. In this project, I built a time-series forecasting model using a Transformer architecture to predict weather conditions based on historical weather data from the previous year.

## **Data Collection**
* The data used to build this foundational model is taken from ERA5 website. Data is from year 2013 to 2025. Usually the data is in csv or excel format but here data is in a special format called "grib" format, GRIB is a machine-optimized format designed to efficiently store huge, multi-dimensional weather datasets such as temperature, wind, pressure, humidity, rainfall, etc., over latitude–longitude grids, multiple heights, and many time steps

* Columns in dataset are time, latitude, longitude, valid time, temperature. 

* The data set is quite big for every year I have around 12 crore datapoints in total.

## **Data Processing**
* In any time series data for unique identification of any time series their is an Id, in my case this unique id is combination of latitude and longitude i.e "<lat,long>". Their are total 14625 unique Ids.

* Unwanted columns are removed and precision is been reduced so that size of such big data is reduced, along with that files are stored in different parts, as I have 14625 Ids I divided this whole data into 5 files each containing 2925 Ids.

