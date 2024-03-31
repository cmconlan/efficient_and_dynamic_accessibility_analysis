import csv
import numpy as np
import itertools
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import os
from typing import Tuple
import time



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
    print('Next Process : {}'.format(process_id))
    output_file = os.path.join(output_dir, f'temp_{process_id}.csv')
    bad_rows = 0

    first_row_found = False

    with open(input_file, 'r') as csv_file:
        reader = csv.DictReader(csv_file)
        csv_section = get_csv_section(reader, offset, limit)
        firstRow = next(csv_section)
        while first_row_found == False:
            t0 = time.time()
            first_response = get_otp_response(host_url, firstRow)
            t1 = time.time()
            if first_response:
                first_response['queryTime'] = t1 - t0
                headers = first_response.keys()
                first_row_found = True
            else:
                bad_rows += 1
                firstRow = next(csv_section)
        
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