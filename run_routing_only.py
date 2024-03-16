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

def main():

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
            'endHour' : 18,
            'endMinute' : 00
            },
        'bh':{
            'startHour' : 10,
            'startMinute' : 00,
            'endHour' : 18,
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
    wm_oas = wm_oas[wm_oas['LAD11CD'] == 'E08000026']
    oa_info = pd.read_csv('data/oa_info.csv')
    oa_info = oa_info.merge(wm_oas[['OA11CD']], left_on = 'oa_id', right_on = 'OA11CD', how = 'inner')
    oaLatLon = oa_info[['oa_id','oa_lon','oa_lat']]

    # Get POIs
    pois = pd.read_csv('data/POIs/pois_cov.csv', index_col=0)

    # Generate POI attractiveness using gaussian randomness
    mean = 50
    std_dev = 20
    attractivnessDict = {}

    for pid in list(pois['poi_id']):
        random_value = random.normalvariate(mean, std_dev)
        attractivnessDict[pid] = max(min(random_value, 100), 0)

    # Create tracking matrices
    performance = {}
    processing_times = {}
    exp_meta_data = {}
    for i in range(num_iterations):
        performance[i] = {}
        processing_times[i] = {}
        exp_meta_data[i] = {}
        for k in list(stratumDict.keys()):
            performance[i][k] = {}
            processing_times[i][k] = {}
            exp_meta_data[i][k] = {}

    for it in range(num_iterations):
        #Sample 200 random zones
        oaSample = oa_info.sample(num_oas)[['oa_id','oa_lat','oa_lon']]
        #POIs
        POISample = pois.sample(num_pois)
        POISample['attractiveness'] = POISample['poi_id'].map(attractivnessDict)
        for stratum in stratumDict.keys():
            
            print('-----NEXT IT ----')
            print('Overall Iteration {} of {}'.format(it,num_iterations))
            print('Stratum : {}'.format(stratum))

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

            exp_meta_data[it][stratum]['oas'] = oaSample
            exp_meta_data[it][stratum]['pois'] = POISample
            exp_meta_data[it][stratum]['time_domain'] = timeDomain

            # Output trips to CSV

            temp_trips_file = 'tempdata/trips_to_route_{}_{}.csv'.format(it,stratum)
            output_file = open(temp_trips_file, 'w')
            writer = csv.writer(output_file)
            writer.writerow(['oa_id','poi_id','trip_id','date','time','oa_lat','oa_lon','poi_lat','poi_lon'])

            #Output trips dataset - output csv with following: 
            trip_id = 0
            #trip_date = study_date[0].strftime('%m/%d/%Y')
            trip_date = study_date[0].strftime('%Y-%m-%d')

            for oind, orow in oaSample.iterrows():
                for pind,prow in POISample.iterrows():
                    for t in timeDomain:            
                        row = [orow['oa_id'],prow['poi_id'],trip_id,trip_date,t,orow['oa_lat'], orow['oa_lon'],prow['poi_lat'], prow['poi_lon']]
                        writer.writerow(row)
                        trip_id += 1

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

            exp_meta_data[it][stratum]['routing_cost'] = t1_routing_cost - t0_routing_cost
            exp_meta_data[it][stratum]['num_trips'] = num_trips
            

            files = []
            bad_rows = 0
            for f, rows in results:
                files.append(f)
                bad_rows += rows
            if bad_rows > 0:
                print(f"{bad_rows} trips were lost during OTP processing ({(bad_rows/num_trips * 100):.2f}%).")

            exp_meta_data[it][stratum]['bad trips'] = bad_rows

            output_file_name = os.path.join('tempdata', 'results_full_{}_{}.csv'.format(it,stratum))
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

            with open('results/exp_meta_data.pkl', 'wb') as f:
                pickle.dump(exp_meta_data, f)

if __name__ == "__main__":
    main()