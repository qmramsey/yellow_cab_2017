# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read .csv to DataFrame
taxi_df = pd.read_csv("C:\\Users\\Huuuge Bitch\\Documents\\My Code Projects\\Code\\SQL\\2017_Yellow_Taxi_Trip_Data.csv")

start_shape = taxi_df.shape

# Rename ID Column
taxi_df.rename(columns= {'Unnamed: 0': "id"}, inplace= True)

# Change Columns with Date info to Datetime
taxi_df['tpep_pickup_datetime'] = pd.to_datetime(taxi_df['tpep_pickup_datetime'])
taxi_df['tpep_dropoff_datetime'] = pd.to_datetime(taxi_df['tpep_dropoff_datetime'])

# Create a new column of ride duration
def insert_ride_duration():
    try:
        taxi_df.insert(loc= 4, column= "ride_duration", value= [(x - y).total_seconds() for (x, y) in zip(taxi_df["tpep_dropoff_datetime"], taxi_df["tpep_pickup_datetime"])])
    # Unless it already exists, then just recalculate it anyway
    except ValueError:
        taxi_df["ride_duration"] = [(x - y).total_seconds() for (x, y) in zip(taxi_df["tpep_dropoff_datetime"], taxi_df["tpep_pickup_datetime"])]
insert_ride_duration()

# One Ride Duration was negative. Switch the dropoff and pickup times and recalculate ride_duration
def swap_duration(df):
    """
    Finds the only row with a negative time and switches the pickup and dropoff times
    """
    row_idx = df[df['id'] == 93542707].index # Find the row in question
    v1, v2, = df['tpep_pickup_datetime'][row_idx], df['tpep_dropoff_datetime'][row_idx] # Take the times
    df['tpep_pickup_datetime'][row_idx], df['tpep_dropoff_datetime'][row_idx] = v2, v1 # Switch and reassign
    return df
taxi_df = swap_duration(taxi_df)
# Now, recalculate the values of ride_duration
insert_ride_duration()

# Individual Variables
# Some of the validation discovered rows that fit multiple criteria

# One RatecodeID is 99. Drop it.
ratecodeid_map = [True if x in [1, 2, 3, 4, 5, 6] else False for x in taxi_df["RatecodeID"]]
# inv_ratecodeid_map = np.invert(ratecodeid_map)
# print(f"Ratecode ID -- Length: {len(taxi_df[inv_ratecodeid_map])}\n", taxi_df[inv_ratecodeid_map])
taxi_df = taxi_df[ratecodeid_map]


# 30 Rides had 0 passengers
# print(f"Passengers -- Length: {len(taxi_df[taxi_df["passenger_count"] == 0])}\n", taxi_df[taxi_df["passenger_count"] == 0])

# No rides had "Unknown" or "Voided Trip" as a payment type.
# 114 had "No Charge" and 39 had "Dispute"
# print(f"Payment Type: No Charge -- Length: {len(taxi_df[taxi_df["payment_type"] == 3])}\n", taxi_df[taxi_df["payment_type"] == 3])
# print(f"Payment Type: Dispute -- Length: {len(taxi_df[taxi_df["payment_type"] == 4])}\n", taxi_df[taxi_df["payment_type"] == 4])

# 14 Rides have negative total_amount. All from vendor 2. Drop them.
# print(f"Fee -- Length: {len(taxi_df[taxi_df["total_amount"] < 0])}\n", taxi_df[taxi_df["total_amount"] < 0]) # shows dropped rows
taxi_df = taxi_df[taxi_df["total_amount"] >= 0]

# A few rides have extremely large fare amounts. Leave for now but consider removing the top several - maybe anything above 100 (13 rides)
# print(f"Fare Amount -- Length: {len(taxi_df[taxi_df["fare_amount"] > 100])}\n", taxi_df[taxi_df["fare_amount"] > 100]) # shows dropped rows
taxi_df = taxi_df[taxi_df["fare_amount"] <= 100]

# Extra can only be 0.5 or 1 for the rush-hour or overnight charges or 0 for no charge.
# 110 rides have a differnt value (most also have a fare amount of $52). For now, I will drop them.
extra_map = [True if x in [0, 0.5, 1] else False for x in taxi_df['extra']]
# inv_extra_map = np.invert(extra_map)
# print(f"Ratecode ID -- Length: {len(taxi_df[inv_extra_map])}\n", taxi_df[inv_extra_map]) # shows dropped rows
taxi_df = taxi_df[extra_map]

# MTA Tax can only be 0.5. Drop the rest.
# 103 Rides have an MTA tax that is 0 or negative
# print(f"MTA Tax -- Length: {len(taxi_df[taxi_df["mta_tax"] != 0.5])}\n", taxi_df[taxi_df["mta_tax"] != 0.5]) # shows dropped rows
taxi_df = taxi_df[taxi_df["mta_tax"] == 0.5]

# Improvement Surcharge has to be 0.3 (Changed in 2015). Drop all that are not
# 20 are not. Some are 0, others are negative.
# print(f"Surcharge -- Length: {len(taxi_df[taxi_df["improvement_surcharge"] != 0.3])}\n", taxi_df[taxi_df["improvement_surcharge"] != 0.3]) # shows dropped rows
taxi_df = taxi_df[taxi_df["improvement_surcharge"] == 0.3]

# Tip amounts are usually small, but two rides had tips > 40
# print(f"Tip -- Length: {len(taxi_df[taxi_df["tip_amount"] > 40])}\n", taxi_df[taxi_df["tip_amount"] > 40])
# taxi_df = taxi_df[taxi_df["tip_amount"] <= 40]

# Tolls amounts are usually quite low, but none are more than $20
# 1174 rides left a tip amount > $0.
# print(f"Tip -- Length: {len(taxi_df[taxi_df["tolls_amount"] > 0])}\n", taxi_df[taxi_df["tolls_amount"] > 0])
# taxi_df = taxi_df[taxi_df["tolls_amount"] == 0]

# Ride Duration
def ride_duration_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates and removes outliers from the dataframe given in column "ride_duration"
    """
    # Removes 1227 Rows of outliers
    upper = np.percentile(df["ride_duration"], 75)
    lower = np.percentile(df["ride_duration"], 25)
    iqr = (upper - lower)
    # print(f"Low: {lower - (1.5 * iqr)}\nHigh: {upper + (1.5 * iqr)}")
    df = df[df['ride_duration'] >= lower - (1.5 * iqr)] # Keep if above -654.5
    df = df[df['ride_duration'] <= upper + (1.5 * iqr)] # Keep if below 2157.5 (35 min, 57.5 sec)
    return df
taxi_df = ride_duration_outliers(taxi_df)

# Distance Travelled
def distance_travelled_outliers(df: pd.DataFrame) -> pd.DataFrame: # Won't run because values are reasonable, even if there are outliers.
    """
    Calculates and removes outliers from distance_travelled
    """
    # Removes 1952 rows of outliers
    upper = np.percentile(df["trip_distance"], 75)
    lower = np.percentile(df["trip_distance"], 25)
    iqr = (upper - lower)
    # print(f"Low: {lower - (1.5 * iqr)}\nHigh: {upper + (1.5 * iqr)}")
    df = df[df['trip_distance'] >= lower - (1.5 * iqr)] # Keep if above -1.7
    df = df[df['trip_distance'] <= upper + (1.5 * iqr)] # Keep if below 5.34
    return df
# taxi_df = distance_travelled_outliers(taxi_df)

# Fare Amount
def fare_amount_outliers(df: pd.DataFrame) -> pd.DataFrame: # Won't run because values are reasonable, even if there are outliers.
    """
    Calculates and removes outliers from fare_amount
    """
    # Removes 1323 rows of outliers
    upper = np.percentile(df["fare_amount"], 75)
    lower = np.percentile(df["fare_amount"], 25)
    iqr = (upper - lower)
    print(f"Low: {lower - (1.5 * iqr)}\nHigh: {upper + (1.5 * iqr)}")
    df = df[df['fare_amount'] >= lower - (1.5 * iqr)] # Keep if above -4.0
    df = df[df['fare_amount'] <= upper + (1.5 * iqr)] # Keep if below 24.0
    return df
# taxi_df = fare_amount_outliers(taxi_df)

# Tips
def tip_amount_outliers(df: pd.DataFrame) -> pd.DataFrame: # Won't Run because values for tips are reasonable
    """
    Calculates and removes outliers from tip_amount
    """
    # Removes 1323 rows of outliers
    upper = np.percentile(df["tip_amount"], 75)
    lower = np.percentile(df["tip_amount"], 25)
    iqr = (upper - lower)
    print(f"Low: {lower - (1.5 * iqr)}\nHigh: {upper + (1.5 * iqr)}")
    s = df.shape[0]
    df = df[df['tip_amount'] >= lower - (1.5 * iqr)] # Keep if above -3.39
    m = df.shape[0]
    df = df[df['tip_amount'] <= upper + (1.5 * iqr)] # Keep if below 5.65
    e = df.shape[0]
    print(s, m, e, sep='\n')
    return df
# taxi_df = tip_amount_outliers(taxi_df)

end_shape = taxi_df.shape
print(f"The df was reduced by {start_shape[0] - end_shape[0]} rows after validation and removing outliers")

# Quick plots
def make_boxplot(x):
    fig = plt.figure()
    plt.boxplot(x= x)

    plt.xlabel("Distribution")
    plt.title("Boxplot of Variable")

    plt.show()


def make_barplot(s: pd.Series):
    fig = plt.figure()
    vc = s.value_counts()
    plt.bar(x = vc.index, height= vc)

    plt.xlabel("Value Count Index")
    plt.title("Barplot of Variable")

    plt.show()

def make_histogram(x):
    fig = plt.figure()
    plt.hist(x= x, bins= 30)

    plt.xlabel("X-Variable")
    plt.ylabel("Number / Count")
    plt.title("Histogram of the Count of X-Variable")

    plt.show()

# print(taxi_df.columns)
# make_histogram(taxi_df["ride_duration"])
# make_barplot(taxi_df["passenger_count"])
# make_barplot(taxi_df["payment_type"])
# make_histogram(taxi_df['fare_amount'])
# make_histogram(taxi_df['tip_amount'])
# make_boxplot(taxi_df["tolls_amount"])

def make_csv():
    taxi_df.to_csv("yellow_taxi_data_2017_cleaned.csv", index= False)
# make_csv()