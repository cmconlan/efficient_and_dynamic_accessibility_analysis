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


            initial_trips = pd.read_csv('tempdata/trips_to_route.csv').set_index('trip_id')
            trips = pd.read_csv('tempdata/results_full.csv').set_index('trip_id')
            trips = trips.merge(initial_trips[['poi_id','oa_id','time']],right_index=True, left_index=True)

            exp_meta_data[it][stratum]['num_trips_initial'] = len(initial_trips)
            exp_meta_data[it][stratum]['num_trips_costed'] = len(trips)

            #OA-POI Index Dict
            oa_poi_id_dict = {}

            for oa_id in list(oaSample['oa_id']):
                oa_poi_id_dict[oa_id] = {}
                for poi_id in list(POISample['poi_id']):
                    oa_poi_id_dict[oa_id][poi_id] = list(trips[(trips['oa_id'] == oa_id) & (trips['poi_id'] == poi_id)].index)

            peformance_mx = pd.DataFrame(index = list(oaSample['oa_id']))
            costing_mx = pd.DataFrame(index = list(oaSample['oa_id']))
            computing_times = {}

            #Compute generalised access cost
            trips['gac'] = (( 1.5 * (trips['total_time'])) - (0.5 * trips['transit_time']) + ((trips['fare'] * 3600) / 6.7) + (10 * trips['num_transfers'])) / 60
            trips['att'] = trips['poi_id'].map(attractivnessDict)

            #Gravity Model Ground Truth
            trips['dist decay'] = vfunc(np.array(trips['gac']),decay_constant, exponent)
            trips['grav'] = trips['dist decay'] * trips['att']
            gravity = trips.groupby('oa_id').sum()['grav']

            peformance_mx['GM'] = gravity
            costing_mx['GM'] = trips.groupby('oa_id').sum()['queryTime']

            # Run KNN

            for k in [5,15,25]:

                processing_time = 0

                t0 = time.time()

                knn = NearestNeighbors(n_neighbors=k)
                knn.fit(POISample[['poi_lon','poi_lat']].values)
                kResList = []
                countOrigins = 0

                t1 = time.time()
                processing_time += (t1 - t0)

                for oind, orow in oaSample.iterrows():
                    t0 = time.time()
                    distances, indices = knn.kneighbors(orow[['oa_lon','oa_lat']].values.reshape(1, -1))
                    t1 = time.time()
                    processing_time += (t1 - t0)
                    
                    oCosts = []
                    oTrips = 0
                    oTimes = 0

                    for i in indices[0]:
                        pid = POISample.iloc[i]['poi_id']
                        #trips_sample = trips[(trips['oa_id'] == orow['oa_id']) & (trips['poi_id'] == pid)]
                        trips_sample = trips.loc[oa_poi_id_dict[orow['oa_id']][pid]]
                        oCosts = oCosts + list(trips_sample['gac'])
                        oTrips += len(trips_sample)
                        oTimes += trips_sample['queryTime'].sum()

                    kResAppend = {}
                    kResAppend['oa_id'] = orow['oa_id']
                    kResAppend['score'] = statistics.mean(oCosts)
                    kResAppend['times'] = oTimes
                    kResList.append(kResAppend)

                knnres = pd.DataFrame(kResList).set_index('oa_id')

                peformance_mx['knn_{}'.format(k)] = knnres['score']
                costing_mx['KNN_{}'.format(k)] = knnres['times']
                computing_times['KNN_{}'.format(k)] = processing_time
            #Run k-means

            for num_clusters in [3,5,7,9]:

                processing_time = 0

                t0 = time.time()
                kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init="auto").fit(POISample[['poi_lon','poi_lat']].values)
                # Get cluster labels and centroids
                cluster_labels = kmeans.labels_
                t1 = time.time()
                processing_time += (t1 - t0)

                POISample['cluster'] = cluster_labels

                flows_df_list = []

                for cluster in set(cluster_labels):
                    t0 = time.time()
                    points_in_cluster = np.where(cluster_labels == cluster)[0]
                    centroid = [POISample[POISample['cluster'] == cluster]['poi_lon'].mean(),POISample[POISample['cluster'] == cluster]['poi_lat'].mean()]
                    distances_to_centroid = np.linalg.norm(POISample[['poi_lon','poi_lat']].values[points_in_cluster] - POISample[['poi_lon','poi_lat']].values[points_in_cluster].mean(axis = 0), axis=1)
                    # Find the index of the point closest to the centroid
                    closest_point_index = points_in_cluster[np.argmin(distances_to_centroid)]
                    flow_id = POISample.iloc[closest_point_index]['poi_id']
                    t1 = time.time()
                    processing_time += (t1 - t0)
                    oa_count = 0
                    for oa in list(oaSample['oa_id']):
                        oa_count += 1
                        flow_gac = trips.loc[oa_poi_id_dict[oa][flow_id]][['oa_id','poi_id','time','gac','queryTime']]
                        query_times = list(flow_gac['queryTime'])

                        for poi in points_in_cluster:
                            flow_append = flow_gac.copy()
                            if poi != closest_point_index:
                                flow_append['queryTime'] = 0
                            else:
                                flow_append['queryTime'] = query_times
                            flow_append['poi_id'] = POISample.iloc[poi]['poi_id']
                            flows_df_list.append(flow_append)

                flows_df = pd.concat(flows_df_list, ignore_index=True)
                flows_df['att'] = flows_df['poi_id'].map(attractivnessDict)
                flows_df['dist decay'] = vfunc(np.array(flows_df['gac']),decay_constant, exponent)
                flows_df['grav'] = flows_df['dist decay'] * flows_df['att']
                peformance_mx['kmean_{}'.format(num_clusters)] = flows_df.groupby('oa_id').sum()['grav']
                costing_mx['kmean_{}'.format(num_clusters)] = flows_df.groupby('oa_id').sum()['queryTime']
                computing_times['kmean_{}'.format(num_clusters)] = processing_time

            #DBSCAN

            eps_test = [0.01,0.02,0.03]
            min_samples = [1,3,5]

            for ep in eps_test:
                for ms in min_samples:
                    processing_time = 0

                    t0 = time.time()
                    dbscan = DBSCAN(eps=ep, min_samples=ms).fit(POISample[['poi_lon','poi_lat']].values)

                    # Get cluster labels and centroids
                    cluster_labels = dbscan.labels_
                    POISample['cluster'] = cluster_labels
                    t1 = time.time()
                    processing_time += (t1 - t0)
                    flows_df_list = []

                    for cluster in set(cluster_labels):
                        if cluster == -1:
                            for c_ind in np.where(cluster_labels == cluster)[0]:
                                flow_id = POISample.iloc[c_ind]['poi_id']

                                for oa in list(oaSample['oa_id']):
                                    flow_gac = trips.loc[oa_poi_id_dict[oa][flow_id]][['oa_id','poi_id','time','gac','queryTime']]
                                    flows_df_list.append(flow_gac)
                        else:
                            t0 = time.time()
                            points_in_cluster = np.where(cluster_labels == cluster)[0]
                            centroid = [POISample[POISample['cluster'] == cluster]['poi_lon'].mean(),POISample[POISample['cluster'] == cluster]['poi_lat'].mean()]
                            distances_to_centroid = np.linalg.norm(POISample[['poi_lon','poi_lat']].values[points_in_cluster] - POISample[['poi_lon','poi_lat']].values[points_in_cluster].mean(axis = 0), axis=1)
                            # Find the index of the point closest to the centroid
                            closest_point_index = points_in_cluster[np.argmin(distances_to_centroid)]
                            flow_id = POISample.iloc[closest_point_index]['poi_id']
                            t1 = time.time()
                            processing_time += (t1 - t0)
                            for oa in list(oaSample['oa_id']):
                                flow_gac = trips.loc[oa_poi_id_dict[oa][flow_id]][['oa_id','poi_id','time','gac','queryTime']]
                                query_times = list(flow_gac['queryTime'])
                                for poi in points_in_cluster:
                                    flow_append = flow_gac.copy()
                                    if poi != closest_point_index:
                                        flow_append['queryTime'] = 0
                                    else:
                                        flow_append['queryTime'] = query_times
                                    flow_append['poi_id'] = POISample.iloc[poi]['poi_id']
                                    flows_df_list.append(flow_append)

                    flows_df = pd.concat(flows_df_list, ignore_index=True)
                    flows_df['att'] = flows_df['poi_id'].map(attractivnessDict)
                    flows_df['dist decay'] = vfunc(np.array(flows_df['gac']),decay_constant, exponent)
                    flows_df['grav'] = flows_df['dist decay'] * flows_df['att']
                    peformance_mx['dbscan_{}_{}'.format(ep,ms)] = flows_df.groupby('oa_id').sum()['grav']
                    costing_mx['dbscan_{}_{}'.format(ep,ms)] = flows_df.groupby('oa_id').sum()['queryTime']
                    computing_times['dbscan_{}_{}'.format(ep,ms)] = processing_time

            #HDBSCAN

            min_clusters = [3,5,7]
            min_sample_tests = [1,3,5,7]

            for mc in min_clusters:
                for ms in min_sample_tests:

                    processing_time = 0

                    t0 = time.time()
                    hdbscan = HDBSCAN(min_cluster_size = mc, min_samples=ms).fit(POISample[['poi_lon','poi_lat']].values)

                    # Get cluster labels and centroids
                    cluster_labels = hdbscan.labels_
                    POISample['cluster'] = cluster_labels
                    t1 = time.time()
                    processing_time += (t1 - t0)

                    flows_df_list = []

                    for cluster in set(cluster_labels):
                        if cluster == -1:
                            for c_ind in np.where(cluster_labels == cluster)[0]:
                                flow_id = POISample.iloc[c_ind]['poi_id']

                                for oa in list(oaSample['oa_id']):
                                    flow_gac = trips.loc[oa_poi_id_dict[oa][flow_id]][['oa_id','poi_id','time','gac','queryTime']]
                                    flows_df_list.append(flow_gac)
                        else:
                            t0 = time.time()
                            points_in_cluster = np.where(cluster_labels == cluster)[0]
                            centroid = [POISample[POISample['cluster'] == cluster]['poi_lon'].mean(),POISample[POISample['cluster'] == cluster]['poi_lat'].mean()]
                            distances_to_centroid = np.linalg.norm(POISample[['poi_lon','poi_lat']].values[points_in_cluster] - POISample[['poi_lon','poi_lat']].values[points_in_cluster].mean(axis = 0), axis=1)
                            # Find the index of the point closest to the centroid
                            closest_point_index = points_in_cluster[np.argmin(distances_to_centroid)]
                            flow_id = POISample.iloc[closest_point_index]['poi_id']
                            t1 = time.time()
                            processing_time += (t1 - t0)
                            for oa in list(oaSample['oa_id']):
                                flow_gac = trips.loc[oa_poi_id_dict[oa][flow_id]][['oa_id','poi_id','time','gac','queryTime']]
                                query_times = list(flow_gac['queryTime'])
                                for poi in points_in_cluster:
                                    flow_append = flow_gac.copy()
                                    if poi != closest_point_index:
                                        flow_append['queryTime'] = 0
                                    else:
                                        flow_append['queryTime'] = query_times
                                    flow_append['poi_id'] = POISample.iloc[poi]['poi_id']
                                    flows_df_list.append(flow_append)

                    flows_df = pd.concat(flows_df_list, ignore_index=True)
                    flows_df['att'] = flows_df['poi_id'].map(attractivnessDict)
                    flows_df['dist decay'] = vfunc(np.array(flows_df['gac']),decay_constant, exponent)
                    flows_df['grav'] = flows_df['dist decay'] * flows_df['att']
                    peformance_mx['hdbscan_{}_{}'.format(mc,ms)] = flows_df.groupby('oa_id').sum()['grav']
                    costing_mx['hdbscan_{}_{}'.format(mc,ms)] = flows_df.groupby('oa_id').sum()['queryTime']
                    computing_times['hdbscan_{}_{}'.format(mc,ms)] = processing_time


            # Get flow
            # Cluster flows

            minflows = [5,7]
            minclusters = [5,15]
            minsamples = [5,15]

            for mf in minflows:
                t0 = time.time()
                X, o_index, d_index, flows, flows_index = get_flow_dist_mx(oaSample,POISample,mf,flowProx)
                t1 = time.time()
                get_flows_time = t1 - t0
                for mc in minclusters:
                    for ms in minsamples:
                        
                        processing_time = 0

                        t0 = time.time()
                        hdb = HDBSCAN(min_cluster_size=mc, min_samples=ms, metric=getminreach).fit(X)
                        cluster_labels = hdb.labels_
                        t1 = time.time()
                        processing_time += (t1 - t0)
                        flows_df_list = []
                        for cluster in set(cluster_labels):
                            if cluster == -1:
                                cluster_indeces = np.where(cluster_labels==cluster)[0]
                                flow_inds = list(set([flows_index[i] for i in cluster_indeces]))
                                for f in flow_inds:
                                    flow_gac = trips.loc[oa_poi_id_dict[f[0]][f[1]]][['oa_id','poi_id','time','gac','queryTime']]
                                    flows_df_list.append(flow_gac)
                            else:
                                t0 = time.time()
                                cluster_indeces = np.where(cluster_labels==cluster)[0]
                                #Select origin of best flow
                                oas_in_flow = list(set([o_index[i] for i in cluster_indeces]))
                                distances_to_centroid = np.linalg.norm(oaSample.set_index('oa_id').loc[oas_in_flow].values - oaSample.set_index('oa_id').loc[oas_in_flow].values.mean(axis = 0),axis=1)
                                flow_oa = oas_in_flow[np.argmin(distances_to_centroid)]
                                #Select destination of best flow
                                poi_ids = list(set([d_index[i] for i in cluster_indeces]))
                                distances_to_centroid = np.linalg.norm(POISample.set_index('poi_id').loc[poi_ids][['poi_lon','poi_lat']].values - POISample.set_index('poi_id').loc[poi_ids][['poi_lon','poi_lat']].values.mean(axis = 0),axis = 1)
                                flow_poi = poi_ids[np.argmin(distances_to_centroid)]
                                t1 = time.time()
                                processing_time += (t1 - t0)
                                #measure GAC for all time steps
                                flow_gac = trips.loc[oa_poi_id_dict[flow_oa][flow_poi]][['oa_id','poi_id','time','gac','queryTime']]
                                query_times = list(flow_gac['queryTime'])
                                flow_inds = list(set([flows_index[i] for i in cluster_indeces]))
                                for f in flow_inds:
                                    flow_append = flow_gac.copy()
                                    if f[1] != flow_poi:
                                        flow_append['queryTime'] = 0
                                    else:
                                        flow_append['queryTime'] = query_times
                                    flow_append['oa_id'] = f[0]
                                    flow_append['poi_id'] = f[1]
                                    flows_df_list.append(flow_append)
                        flows_df = pd.concat(flows_df_list, ignore_index=True)
                        flows_df['att'] = flows_df['poi_id'].map(attractivnessDict)
                        flows_df['dist decay'] = vfunc(np.array(flows_df['gac']),decay_constant, exponent)
                        flows_df['grav'] = flows_df['dist decay'] * flows_df['att']
                        peformance_mx['flowhdbscan_{}_{}_{}'.format(mf,mc,ms)] = flows_df.groupby('oa_id').sum()['grav']
                        costing_mx['flowhdbscan_{}_{}_{}'.format(mf,mc,ms)] = flows_df.groupby('oa_id').sum()['queryTime']
                        computing_times['flowhdbscan_{}_{}_{}'.format(mf,mc,ms)] = processing_time
                        computing_times['get_flow_flowhdbscan_{}_{}_{}'.format(mf,mc,ms)] = get_flows_time


            # Gravit Trip Generator

            distMxList = []

            for i,r in oaSample.iterrows():
                for i_, r_ in POISample.iterrows():
                    rowAppend = {}
                    rowAppend['oa'] = r['oa_id']
                    rowAppend['poi'] = r_['poi_id']
                    rowAppend['dist'] = haversine_distance(r['oa_lon'], r['oa_lat'], r_['poi_lon'], r_['poi_lat'])
                    distMxList.append(rowAppend)

            distMx = pd.DataFrame(distMxList)
            distMx['att'] = distMx['poi'].map(attractivnessDict)

            distsDecay = []
            for i in np.array(distMx['dist']):
                distsDecay.append(distance_decay(i, decay_constant, exponent))

            distMx['decay'] = distsDecay
            distMx['grav'] = distMx['decay'] * distMx['att']
            distMx['gravN'] = (distMx['grav'] - distMx['grav'].min()) / (distMx['grav'].max() - distMx['grav'].min())

            distMx = distMx.merge(trips.groupby(['oa_id','poi_id']).count()['departure_time'].rename('tripCount'), left_on = ['oa','poi'],right_index = True)
            distMx['tripsSample'] = distMx['tripCount'] * distMx['gravN']

            resultsList = []
            countOrigins = 0

            for o in list(oaSample['oa_id']):
                countOrigins += 1
                oCosts = []
                oTrips = 0
                oTimes = 0
                for p in list(POISample['poi_id']):
                    if len(oa_poi_id_dict[o][p]) > 0:
                        numSample = int(distMx[(distMx['oa'] == o) & (distMx['poi'] == p)]['tripsSample'].values[0])
                        tripSample = trips.loc[oa_poi_id_dict[o][p]].sample(numSample)
                        oCosts = oCosts + list(tripSample['gac'])
                        oTrips += numSample
                        oTimes += tripSample['queryTime'].sum()
                rowAppend = {}
                rowAppend['OA'] = o
                rowAppend['GTG'] = statistics.mean(oCosts)
                rowAppend['GTG_Trips'] = oTrips
                rowAppend['GTG_Times'] = oTimes
                resultsList.append(rowAppend)

            tripGenGrav = pd.DataFrame(resultsList).set_index('OA')

            peformance_mx['g-tgm'] = tripGenGrav['GTG']
            costing_mx['g-tgm'] = tripGenGrav['GTG_Times']

            performance[it][stratum] = peformance_mx
            processing_times[it][stratum] = costing_mx
            exp_meta_data[it][stratum]['processing_times'] = computing_times

            #Delete all file in tempdata
            files = glob.glob('tempdata/*')
            for f in files:
                os.remove(f)

            with open('results/performance.pkl', 'wb') as f:
                pickle.dump(performance, f)

            with open('results/processing_times.pkl', 'wb') as f:
                pickle.dump(processing_times, f)

            with open('results/exp_meta_data.pkl', 'wb') as f:
                pickle.dump(exp_meta_data, f)

if __name__ == "__main__":
    main()