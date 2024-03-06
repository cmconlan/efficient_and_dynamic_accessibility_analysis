import pandas as pd
import geopandas as gpd
from datetime import datetime, timedelta
import random
import numpy as np
import csv
import multiprocessing
import os
from typing import Tuple
import time
import itertools
import requests
import xml.etree.ElementTree as ET

def num_rows(file_name: str) -> int:
    '''Count the number of rows in the (CSV) file'''
    with open(file_name, 'r') as otp_trips:
        otp_trips_reader = csv.reader(otp_trips)
        rows = sum(1 for row in otp_trips_reader)
    rows -= 1 # Account for the header
    return rows

def get_step_size(num_trips: int, processes: int) -> int:
    '''Calculate how many rows each process will have'''
    return int(np.ceil(num_trips / processes))

def init(trips, complete):
    # Make num_trips global in each process.
    # This grants read-only access in compute_trips
    global num_trips
    global rows_complete
    num_trips = trips
    rows_complete = complete

def get_csv_section(reader, offset, limit) -> object:
    '''Get a 'slice' or section of the CSV file for a process'''
    return itertools.islice(reader, offset, limit)

def request_otp(host_url, input_row):
    url = host_url + '/otp/routers/default/plan?'
    params = {
        "fromPlace": f"{input_row['oa_lat']},{input_row['oa_lon']}",
        "toPlace": f"{input_row['poi_lat']},{input_row['poi_lon']}",
        "date": f"{input_row['date']}",
        "time": f"{input_row['time']}",
        "mode": "TRANSIT,WALK",
        "arriveBy": "false",
        "numItineraries": "1",
        #"maxWalkDistance": "1000"
        "walkReluctance": "20"
    }
    resp = requests.get(
        url=url,
        params=params,
        headers={'accept': 'application/xml'}
    )
    return resp

def get_request_parameter(node: ET.Element, param: str) -> str:
    request_parameters = node.find('requestParameters')
    return request_parameters.find(param).text

def get_time_from_itinerary(time: str, itinerary):
    dt = datetime.fromtimestamp(float(itinerary.find(time).text) / 1000)
    dt += timedelta(hours=1)
    return dt

def get_total_distance_from_itinerary(itinerary):
    total_dist = 0.0
    for legs in itinerary.findall('legs'):
        if legs.find('legs') is not None:
            for leg in legs.findall('legs'):
                total_dist += float(leg.find('distance').text)
    return total_dist

def get_fare_from_itinerary(itinerary):
    if itinerary.find('fare') is not None:
        fare_obj = itinerary.find('fare')
        if fare_obj.find('details') is not None:
            return float(fare_obj.find('details').find('regular').find('price').find('cents').text) / 100


def calculate_fare(num_transfers, walk_time, total_time):
    if walk_time == total_time:
        return 0.0
    else:
        return 2.40 * (num_transfers + 1)


def validate_trip(trip: dict) -> bool:
    for value in trip.values():
        if value is None:
            return False
    return True

def parse_response(response):
    root = ET.fromstring(response.content)
    trip = {
        'departure_time': None,
        'arrival_time': None,
        'total_time': None,
        'walk_time': None,
        'transfer_wait_time': None,
        'transit_time': None,
        'walk_dist': None,
        'transit_dist': None,
        'total_dist': None,
        'num_transfers': None,
        'initial_wait_time': None,
        'fare': None
    }
    date = get_request_parameter(root, 'date')
    time = get_request_parameter(root, 'time')
    query_time = datetime.strptime(' '.join([date, time]), '%Y-%m-%d %H:%M')
    # Check if there was an error in the OTP response
    trip_valid = False
    if root.find('error').find('msg') is not None:
        # The start and destination were too close, no trip could be found
        if root.find('error').find('message').text in "TOO_CLOSE":
            trip['departure_time'] = query_time
            trip['arrival_time'] = query_time
            trip['total_time'] = 0.0
            trip['walk_time'] = 0.0
            trip['transfer_wait_time'] = 0.0
            trip['transit_time'] = 0.0
            trip['walk_dist'] = 0.0
            trip['transit_dist'] = 0.0
            trip['total_dist'] = 0.0
            trip['num_transfers'] = 0
            trip['initial_wait_time'] = 0.0
            trip['fare'] = 0.0
            trip_valid = True
    else:
        plan = root.find('plan')
        # Go through the iteneraries found in the plan. Should only be 1
        for itineraries in plan.findall('itineraries'):
            if itineraries.find('itineraries') is not None:  # Note that this line discards error XML, where there was no route
                for itinerary in itineraries.findall('itineraries'):
                    format_str = '%Y-%m-%d %H:%M:%S'
                    trip['arrival_time'] = get_time_from_itinerary('endTime', itinerary).strftime(format_str)
                    trip['total_time'] = float(itinerary.find('duration').text)
                    trip['walk_time'] = float(itinerary.find('walkTime').text)
                    trip['transfer_wait_time'] = float(itinerary.find('waitingTime').text)
                    trip['transit_time'] = float(itinerary.find('transitTime').text)
                    trip['walk_dist'] = float(itinerary.find('walkDistance').text)
                    trip['num_transfers'] = int(itinerary.find('transfers').text)
                    trip['total_dist'] = get_total_distance_from_itinerary(itinerary)
                    trip['fare'] = get_fare_from_itinerary(itinerary)
                    if trip['fare'] is None:
                        trip['fare'] = calculate_fare(trip['num_transfers'], trip['walk_time'], trip['total_time'])
                    # capture the wait time before the first bus arrives
                    departure_time = get_time_from_itinerary('startTime', itinerary)
                    trip['initial_wait_time'] = (departure_time - query_time).total_seconds()
                    trip['departure_time'] = departure_time.strftime(format_str)
                    trip['transit_dist'] = trip['total_dist'] - trip['walk_dist']
                    trip_valid = validate_trip(trip)
    if trip_valid:
        return trip
    else:
        return False

def compute_trips(host_url: str, offset: int, limit: int, input_file: str, output_dir: str) -> Tuple[str, int]:
    """
    Send a request to OTP, parse the response and write a line to the output file.
    Note: Parallel processing begins and ends here - each Python process will run this
    function until it has completed its 'chunk' of data, then it will return the name
    of the file it wrote its data to.
    Recall that individual processes do not share global variables and other data - each 
    process holds a copy of the parent's (process it was created from) data independently 
    of any other process.
    """
    process_id = os.getpid()
    output_file = os.path.join(output_dir, f'temp_{process_id}.csv')
    bad_rows = 0
    with open(input_file, 'r') as csv_file:
        reader = csv.DictReader(csv_file)
        csv_section = get_csv_section(reader, offset, limit)
        firstRow = next(csv_section)
        t0 = time.time()
        first_response = get_otp_response(host_url, firstRow)
        t1 = time.time()
        first_response['queryTime'] = t1 - t0
        headers = first_response.keys()
        
        with open(output_file, 'a', newline='') as output_csv:
            writer = csv.DictWriter(output_csv, fieldnames=headers, delimiter=',')
            writer.writeheader()
            writer.writerow(first_response)
            row_counter = 1
            for row in csv_section:
                t0 = time.time()
                response = get_otp_response(host_url, row)
                t1 = time.time()
                # Some trips have None for all attributes due to OTP error or inability to find a trip
                # These trips return 'False' instead of a dict so empty rows are not written to CSV.
                row_counter += 1
                if response:
                    response['queryTime'] = t1 - t0
                    writer.writerow(response)
                else:
                    bad_rows += 1
                if row_counter % 1000 == 0:
                    row_counter = 0
    return output_file, bad_rows

def get_otp_response(host_url, input_row) -> tuple:
    '''Parse the response from OTP into tuple of values represnting trip attributes'''
    response = request_otp(host_url, input_row)
    trip = parse_response(response)
    if trip:
        trip['trip_id'] = input_row['trip_id']
    return trip

def extract_headers(csv_file):
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        return next(reader)
    

host = 'localhost'
port = 8080
processes = 12
otps = 2


# Get Time Stamps / Time Interval

stratumDict = {
    'amPeak':{
        'startHour' : 6,
        'startMinute' : 30,
        'endHour' : 9,
        'endMinute' : 00
        },
    'Saturday':{
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

mean = 50
std_dev = 20
attractivnessDict = {}

for pid in list(pois['poi_id']):
    random_value = random.normalvariate(mean, std_dev)
    attractivnessDict[pid] = max(min(random_value, 100), 0)

#Sample 200 random zones
oaSample = oa_info.sample(200)[['oa_id','oa_lat','oa_lon']]

#POIs
POISample = pois.sample(50)
POISample['attractiveness'] = POISample['poi_id'].map(attractivnessDict)

stratum = 'amPeak'

if stratum == 'amPeak':
    study_date = experiment_dates[(experiment_dates['weekday']) & (experiment_dates['bank_holiday'] == False)].sample(1).index

elif stratum == 'Saturday':
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

timeDomain = []

for i in range(50):
    randStartTime = start + timedelta(minutes=random.randint(1, int(minutesInInterval)))
    if randStartTime not in timeDomain:
        timeDomain.append(str(randStartTime.hour).zfill(2)+':'+str(randStartTime.minute).zfill(2))

temp_trips_file = 'tempdata/trips_to_route.csv'
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
print('Time Taken : {}'.format(t1_routing_cost - t0_routing_cost))

files = []
bad_rows = 0
for f, rows in results:
    files.append(f)
    bad_rows += rows
if bad_rows > 0:
    print(f"{bad_rows} trips were lost during OTP processing ({(bad_rows/num_trips * 100):.2f}%).")

output_file_name = os.path.join('tempdata', 'results_full.csv')
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