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
from shapely import Point

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
decay_constant = config['decay_constant']
exponent = config['exponent']
#Integer for 1 trip per x minutes. E.g., 1 = a trip every minutes. 5 = 1 trip every 5 minutes
# For now go with every minutes but may pull this back
# To do - revisit later on
temp_resolution = config['temp_resolution']
num_iterations = config['num_iterations']
num_oas = config['num_oas']
num_pois = config['num_pois']
processes = config['processes']
otps = config['otps']
host = config['host']
port = config['port']

# Vectrise distance decay function
vfunc = np.vectorize(distance_decay)

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
end_date = '2024-04-15'

# Create a date-time index
date_index = pd.date_range(start=start_date, end=end_date, freq='D')
experiment_dates = pd.DataFrame(index = date_index)
experiment_dates['weekday'] = experiment_dates.index.weekday < 5
experiment_dates['saturday'] = experiment_dates.index.weekday == 5
bank_holidays = ['2024-03-29', '2024-04-01']
experiment_dates['bank_holiday'] = experiment_dates.index.isin(pd.to_datetime(bank_holidays))

# Get OAs
wm_oas = gpd.read_file('data/west_midlands_OAs/west_midlands_OAs.shp')
oa_info = pd.read_csv('data/oa_info.csv')
oa_info = oa_info.merge(wm_oas[['OA11CD']], left_on = 'oa_id', right_on = 'OA11CD', how = 'inner')
oaLatLon = oa_info[['oa_id','oa_lon','oa_lat']]


experiment_meta_data = []
for area_lad in ['E08000025','E08000026']:

    if area_lad == 'E08000025':
        pois = pd.read_csv('data/POIs/pois_birm.csv', index_col=0)
    else:
        pois = pd.read_csv('data/POIs/pois_cov.csv', index_col=0)
    trip_oas = oa_info[oa_info['oa_id'].isin(list(wm_oas[wm_oas['LAD11CD'] == area_lad]['OA11CD']))]

    for p_type in ['School','GP Surgery','Vaccination Centre','Hospital','Job Centre']:
        trip_pois = pois[pois['type'] == p_type]
        for stratum in ['wdam','wdpm','sat','bh']:
            num_trips_odt = 0
            num_trips_gtgm = 0
            trip_generation_cost = 0
            next_exp_meta = {}
            next_exp_meta['area'] = area_lad
            next_exp_meta['poi_type'] = p_type
            next_exp_meta['poi_type'] = stratum

            for it in range(5):

                print()
                print()
                print('---- NEXT IT -----')
                print('Area : {}'.format(area_lad))
                print('POI Type : {}'.format(p_type))
                print('Stratum : {}'.format(stratum))
                print('Iteration : {}'.format(it))

                t0 = time.time()

                if stratum == 'wdam' or stratum == 'wdpm' :
                    study_date = experiment_dates[(experiment_dates['weekday']) & (experiment_dates['bank_holiday'] == False)].sample(1).index

                elif stratum == 'sat':
                    study_date = experiment_dates[(experiment_dates['saturday']) & (experiment_dates['bank_holiday'] == False)].sample(1).index

                elif stratum == 'bh':
                    study_date = experiment_dates[experiment_dates['bank_holiday']].sample(1).index

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

                # Gravit Trip Generator
                distMxList = []

                for i,r in trip_oas.iterrows():
                    for i_, r_ in trip_pois.iterrows():
                        rowAppend = {}
                        rowAppend['oa'] = r['oa_id']
                        rowAppend['oa_lat'] = r['oa_lat']
                        rowAppend['oa_lon'] = r['oa_lon']
                        rowAppend['poi'] = r_['poi_id']
                        rowAppend['poi_lat'] = r_['poi_lat']
                        rowAppend['poi_lon'] = r_['poi_lon']
                        rowAppend['dist'] = haversine_distance(r['oa_lon'], r['oa_lat'], r_['poi_lon'], r_['poi_lat'])
                        distMxList.append(rowAppend)

                distMx = pd.DataFrame(distMxList)
                distMx['att'] = 1

                distsDecay = []
                for i in np.array(distMx['dist']):
                    distsDecay.append(distance_decay(i, decay_constant, exponent))

                distMx['decay'] = distsDecay
                distMx['grav'] = distMx['decay'] * distMx['att']
                distMx['gravN'] = (distMx['grav'] - distMx['grav'].min()) / (distMx['grav'].max() - distMx['grav'].min())
                distMx['num_trips'] = (distMx['gravN'] * len(timeDomain)).astype(int)

                num_trips_gtgm += (distMx['num_trips'].sum())
                num_trips_odt += (len(distMx) * len(timeDomain))

                # Generate
                # Output trips to CSV
                temp_trips_file = 'tempdata/trips_to_route_{}_{}_{}.csv'.format(area_lad,p_type,stratum)
                output_file = open(temp_trips_file, 'w')
                writer = csv.writer(output_file)
                writer.writerow(['oa_id','poi_id','trip_id','date','time','oa_lat','oa_lon','poi_lat','poi_lon'])

                #Output trips dataset - output csv with following: 
                trip_id = 0
                #trip_date = study_date[0].strftime('%m/%d/%Y')
                trip_date = study_date[0].strftime('%Y-%m-%d')

                for i,r in distMx.iterrows():
                    sample_trip_time = random.sample(timeDomain, r['num_trips'])
                    for t in sample_trip_time:
                        row = [r['oa'],r['poi'],trip_id,trip_date,t,r['oa_lat'], r['oa_lon'],r['poi_lat'], r['poi_lon']]
                        writer.writerow(row)
                        trip_id += 1

                t1 = time.time()

                trip_generation_cost += (t1 - t0)

            t0 = time.time()
            # Cost trips on OTP using parallelisation
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

            output_file_name = os.path.join('tempdata', 'results_full_{}_{}_{}.csv'.format(area_lad,p_type,stratum))
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
            t1 = time.time()
            
            next_exp_meta['odt_trips'] = num_trips_odt
            next_exp_meta['gtgm_trips'] = num_trips_gtgm
            next_exp_meta['time_cost'] = (t1-t0)
            next_exp_meta['bad rows'] = bad_rows
            experiment_meta_data.append(next_exp_meta)

        exp_meta_data_df = pd.DataFrame(experiment_meta_data)
        exp_meta_data_df.to_csv('ostaat_trips_metadata.csv')