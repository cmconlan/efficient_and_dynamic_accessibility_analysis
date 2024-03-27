from math import radians, cos, sin, atan2, sqrt,exp,pow
from sklearn.neighbors import NearestNeighbors
import numpy as np

def distance_decay(distance, decay_constant, exponent):
    return exp(-decay_constant * pow(distance, exponent))

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the haversine distance between two points given their latitude and longitude.

    Parameters:
    - lat1, lon1: Latitude and longitude of the first point (in degrees)
    - lat2, lon2: Latitude and longitude of the second point (in degrees)

    Returns:
    The haversine distance in kilometers.
    """
    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    # Radius of the Earth in kilometers (mean value)
    earth_radius = 6371.0

    # Calculate the haversine distance
    distance = earth_radius * c

    return distance

def flowProx(flow_i,flow_j, alpha=1, beta=1, gamma=1):
    dist = (alpha*((flow_i[0] - flow_j[0]) ** 2) + beta*(((flow_i[2] - flow_j[2]) ** 2) + ((flow_i[3] - flow_j[3]) ** 2))) / ((haversine_distance(flow_i[0],flow_i[1],flow_i[2],flow_i[3]) * haversine_distance(flow_j[0],flow_j[1],flow_j[2],flow_j[3]))**gamma)
    return dist


def getminreach(xi,xj):
    flow_i=tuple([xi[0],xi[1],xi[2],xi[3]])
    flow_j=tuple([xj[0],xj[1],xj[2],xj[3]])
    CDi = xi[4]
    CDj = xj[4]
    prox_dist = flowProx(flow_i,flow_j, alpha=1, beta=1, gamma=1)
    mreachd = max(CDi,CDj,prox_dist)
    return mreachd

def get_flow_dist_mx(oaSample,POISample,minflows,flowProx):
    flows_index = []
    o_index = []
    d_index = []
    flows = []

    oind = 0
    for o in oaSample[['oa_lat','oa_lon','oa_id']].values:
        dind = 0
        for d in POISample[['poi_lon','poi_lat','poi_id']].values:
            flows.append(tuple([o[0], o[1], d[0], d[1]]))
            flows_index.append(tuple([o[2],int(d[2])]))
            o_index.append(o[2])
            d_index.append(int(d[2]))
            dind += 1
        oind += 1

    # Calculate CoreD
    # Create a NearestNeighbors object with a custom distance function
    neighbors_model = NearestNeighbors(n_neighbors=minflows, algorithm='ball_tree', metric=flowProx)
    X = np.array(flows)
    # Fit the model to your data
    neighbors_model.fit(X)

    core_distances = []
    count = 0
    for i in X:
        distances, indices = neighbors_model.kneighbors([i])
        core_distances.append(distances[:, -1][0])

    lst_reshaped = np.array(core_distances).reshape((len(core_distances), 1))
    return np.concatenate((X, lst_reshaped), axis=1), o_index, d_index, flows, flows_index