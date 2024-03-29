from helper_functions import *
from otp_routing_functions import *
import pandas as pd
import geopandas as gpd
import random
from datetime import datetime, timedelta
import csv
import multiprocessing
import statistics
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import HDBSCAN
import numpy as np
import sys
import os
import glob
import pickle
import yaml
from shapely import Point, box
import osmnx as ox

maxInt = sys.maxsize

while True:
    # decrease the maxInt value by factor 10 
    # as long as the OverflowError occurs.

    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/10)

def init(trips, complete):
    # Make num_trips global in each process.
    # This grants read-only access in compute_trips
    global num_trips
    global rows_complete
    num_trips = trips
    rows_complete = complete


with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)

# Parameters
processes = config['processes']
otps = config['otps']
host = config['host']
port = config['port']

bounding_box = tuple([-1.925354,52.437594,-1.835747,52.491145])

#Get expanded bounding box
bbox_width = abs(bounding_box[0] - bounding_box[2]) /2
print(bbox_width)
bbox_height = abs(bounding_box[1] - bounding_box[3]) / 2
print(bbox_height)
expanded_bbox = tuple([bounding_box[0] - bbox_width,bounding_box[1] - bbox_height,bounding_box[2] + bbox_width,bounding_box[3] + bbox_height])


tags = {"amenity": 'hospital'} # (dict) – Dict of tags used for finding objects in the selected area: pois.
hospitals = ox.features.features_from_bbox(bounding_box[3], bounding_box[1], bounding_box[2], bounding_box[0], tags=tags)

tags = {"healthcare": 'doctor'} # (dict) – Dict of tags used for finding objects in the selected area: pois.
gps = ox.features.features_from_bbox(bounding_box[3], bounding_box[1], bounding_box[2], bounding_box[0], tags=tags) # the table

tags = {"amenity": 'pharmacy'} # (dict) – Dict of tags used for finding objects in the selected area: pois.
pharmacies = ox.features.features_from_bbox(bounding_box[3], bounding_box[1], bounding_box[2], bounding_box[0], tags=tags) # the table

#Combine together

pois_list = []

geometry = []
type = []
subtype = []
poiids = []
poiid = 1

for i,r in hospitals.iterrows():
    if i[0] != 'relation':
        poi_append = {}
        poi_append['poi_id'] = poiid
        poi_append['type'] = 'healthcare'
        poi_append['subtype'] = 'hospital'
        if i[0] == 'node':
            poi_append['geometry'] = [r['geometry'].x,r['geometry'].y]
        else:
            poi_append['geometry'] = [r['geometry'].centroid.x,r['geometry'].centroid.y]
        pois_list.append(poi_append)
        poiid += 1

for i,r in gps.iterrows():
    if i[0] != 'relation':
        poi_append = {}
        poi_append['poi_id'] = poiid
        poi_append['type'] = 'healthcare'
        poi_append['subtype'] = 'surgery'
        if i[0] == 'node':
            poi_append['geometry'] = [r['geometry'].x,r['geometry'].y]
        else:
            poi_append['geometry'] = [r['geometry'].centroid.x,r['geometry'].centroid.y]
        pois_list.append(poi_append)
        poiid += 1

for i,r in pharmacies.iterrows():
    if i[0] != 'relation':
        poi_append = {}
        poi_append['poi_id'] = poiid
        poi_append['type'] = 'healthcare'
        poi_append['subtype'] = 'pharmacy'
        if i[0] == 'node':
            poi_append['geometry'] = [r['geometry'].x,r['geometry'].y]
        else:
            poi_append['geometry'] = [r['geometry'].centroid.x,r['geometry'].centroid.y]
        pois_list.append(poi_append)
        poiid += 1

pois = pd.DataFrame(pois_list)


#Get OAs
oas = gpd.read_file('data/OAs/OA_2021_EW_BGC.shp').to_crs(4326)
bbox = box(bounding_box[0], bounding_box[1], bounding_box[2], bounding_box[3])

ids = []
origin_centroids = []
count = 0
count_found = 0
for i,r in oas.iterrows():
    count += 1
    if r['geometry'].intersects(bbox):
        count_found += 1
        ids.append(r['OA21CD'])
        origin_centroids.append([r['geometry'].centroid.x,r['geometry'].centroid.y])

print('Number found : {}'.format(count_found))
oas_in_study = oas[oas['OA21CD'].isin(ids)]
oas_in_study = oas_in_study.set_index('OA21CD')
oas_in_study['centroid'] = origin_centroids


stratumDict = {
    'wdam':{
        'startHour' : 6,
        'startMinute' : 30,
        'endHour' : 8,
        'endMinute' : 30
        },
    'wdpm':{
        'startHour' : 16,
        'startMinute' : 00,
        'endHour' : 18,
        'endMinute' : 30
        },
    'sat':{
        'startHour' : 10,
        'startMinute' : 00,
        'endHour' : 16,
        'endMinute' : 00
        },
    'bh':{
        'startHour' : 10,
        'startMinute' : 00,
        'endHour' : 16,
        'endMinute' : 00
        }
    }



# Get day index
# Specify the start and end dates
start_date = '2024-03-15'
end_date = '2024-04-22'

# Create a date-time index
date_index = pd.date_range(start=start_date, end=end_date, freq='D')
experiment_dates = pd.DataFrame(index = date_index)
experiment_dates['weekday'] = experiment_dates.index.weekday < 5
experiment_dates['saturday'] = experiment_dates.index.weekday == 5

temp_resolution = 15
for stratum in ['wdam','wdpm','sat']:

    if stratum == 'wdam' or stratum == 'wdpm' :
        study_date = experiment_dates[experiment_dates['weekday']].sample(1).index

    elif stratum == 'sat':
        study_date = experiment_dates[experiment_dates['saturday']].sample(1).index

    # Create Time Domain
    startHour = stratumDict[stratum]['startHour']
    startMinute = stratumDict[stratum]['startMinute']
    endHour = stratumDict[stratum]['endHour']
    endMinute = stratumDict[stratum]['endMinute']

    start = datetime(year=2012, month=2, day=25, hour=startHour, minute = startMinute)
    end = datetime(year=2012, month=2, day=25, hour=endHour, minute = endMinute)
    diff = end - start
    minutesInInterval = diff.total_seconds()/60
    hoursInInterval = minutesInInterval/60

    num_trips = int((60 / temp_resolution) * hoursInInterval)

    timeDomain = []

    for i in range(num_trips):
        randStartTime = start + timedelta(minutes=random.randint(1, int(minutesInInterval)))
        if randStartTime not in timeDomain:
            timeDomain.append(str(randStartTime.hour).zfill(2)+':'+str(randStartTime.minute).zfill(2))

    #trip_date = study_date[0].strftime('%m/%d/%Y')
    trip_date = study_date[0].strftime('%Y-%m-%d')

    temp_trips_file = 'tempdata/trips_to_route_example_{}.csv'.format(stratum)

    output_file = open(temp_trips_file, 'w')
    writer = csv.writer(output_file)
    writer.writerow(['oa_id','poi_id','trip_id','date','time','oa_lat','oa_lon','poi_lat','poi_lon'])
    #Output trips dataset - output csv with following: 
    trip_id = 0

    for i_oa, r_oa in oas_in_study.iterrows():
        for i_poi, r_poi in pois.iterrows():
            for t in timeDomain:
                row = [i_oa,r_poi['poi_id'],trip_id,trip_date,t,r_oa['centroid'][1], r_oa['centroid'][0],r_poi['geometry'][1], r_poi['geometry'][0]]
                writer.writerow(row)
                trip_id += 1


    num_trips = num_rows(temp_trips_file)
    step_size = get_step_size(num_trips, processes)

    args = []
    for i in range(processes):
        host_url = f"http://{host}:{str(port + (i % otps))}"
        offset =i * step_size
        arg = (
            host_url, 
            offset, 
            min(offset+step_size, num_trips), 
            temp_trips_file, 
            'tempdata'
        )
        args.append(arg)

    rows_complete = multiprocessing.Value('i', 0)

    t0_routing_cost = time.time()
    with multiprocessing.Pool(int(processes), initializer=init, initargs=(num_trips, rows_complete)) as pool:
        results = pool.starmap(compute_trips, args)
    t1_routing_cost = time.time()

    files = []
    bad_rows = 0
    for f, rows in results:
        files.append(f)
        bad_rows += rows
    if bad_rows > 0:
        print(f"{bad_rows} trips were lost during OTP processing ({(bad_rows/num_trips * 100):.2f}%).")

    output_file_name = os.path.join('tempdata', 'results_full_example_{}.csv'.format(stratum))
    files_iterator = iter(files)
    # Treat the first file differently, so we can extract headers
    first_file = files[0]
    headers = extract_headers(first_file)
    with open(output_file_name, 'w') as output_file:
        output_csv = csv.writer(output_file)
        output_csv.writerow(headers)
        for csv_file in files:
            with open(csv_file, newline='') as f:
                reader = csv.DictReader(f)
                for line in reader:
                    output_csv.writerow(line.values())
                
                
    #Delete all file in tempdata
    files = glob.glob('tempdata/temp_*')
    for f in files:
        os.remove(f)