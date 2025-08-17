# Databricks notebook source
%pip install folium --quiet
%pip install tabulate --quiet
%pip install catboost --quiet
%pip install optuna --quiet
%pip install optuna-integration[tfkeras] --quiet
%pip install graphframes --quiet

# COMMAND ----------

import tarfile
import os

""" SQL functionsl"""
from pyspark.sql.functions import col, sum as spark_sum, count, desc, asc, year, when, isnan, avg, min, max, hour, round, substring, to_timestamp, date_format, lit, concat
from pyspark.sql.types import DoubleType, FloatType, StringType, IntegerType, DecimalType, TimestampType, DateType, BooleanType
from pyspark.sql.functions import to_timestamp, to_date
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql import SparkSession
from pyspark.sql.functions import percent_rank

""" Spark ML functions""" 
from pyspark.ml.functions import vector_to_array
# from petastorm.tf_utils import make_petastorm_dataset
# from petastorm import make_reader
# from petastorm import make_batch_reader
from pyspark.ml.feature import Imputer
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, RegressionEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.stat import Correlation
from sklearn.compose import ColumnTransformer
from pyspark.ml.classification import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, chi2, RFE, RFECV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.inspection import permutation_importance
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
from xgboost.spark import SparkXGBClassifier 

""" Deep learning imports"""
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import Sequential, layers

""" Data visualization imports"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from folium.plugins import MarkerCluster
from folium.plugins import MiniMap
import matplotlib.cm as cm
from branca.colormap import LinearColormap
import calendar
import branca.colormap as cm
import pyspark.pandas as ps

""" Hyperparameter tuning imports"""
import optuna
from optuna.integration import TFKerasPruningCallback
from sklearn.model_selection import TimeSeriesSplit

""" warning handling"""
import warnings
warnings.filterwarnings('ignore')

"""graph fuctions"""
import networkx as nx
from graphframes import *
from networkx.algorithms import community
from networkx.algorithms.community import greedy_modularity_communities


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #Initial analysis to find the airlines we want to tell our story for - final picked Southwest Airlines

# COMMAND ----------

# Airline Data    
df_flights = spark.read.parquet(f"dbfs:/mnt/mids-w261/datasets_final_project_2022/parquet_airlines_data_3m/")
display(df_flights)

# COMMAND ----------

# Step 1: Total number of rows in the dataset
total_rows = df_flights.count()
print(f"Total rows in dataset: {total_rows}")

# Step 2: Count number of nulls in DEP_DEL15 column
null_count = df_flights.filter(col("DEP_DEL15").isNull()).count()
print(f"Total null values in DEP_DEL15: {null_count}")

# Step 3: Drop null values in DEP_DEL15 column
df_flights_filtered = df_flights.filter(col("DEP_DEL15").isNotNull())

# Step 4: Compute total flights and total delays per airline
df_delays = (
    df_flights_filtered
    .groupBy("OP_UNIQUE_CARRIER")
    .agg(
        count("*").alias("Total_Flights"),  # Count total flights per airline
        F.sum(col("DEP_DEL15")).alias("Total_Delayed_Flights")  # Count delays
    )
    .withColumn("Delay_Percentage", round((col("Total_Delayed_Flights") / col("Total_Flights")) * 100, 2))  # Calculate delay percentage
    .orderBy(desc("Delay_Percentage"))  # Sort by worst performer
)

# Show the top 10 airlines with the highest delay percentage
display(df_delays.limit(10))

# COMMAND ----------

# Airline Data    
df_flights_complete = spark.read.parquet(f"dbfs:/mnt/mids-w261/datasets_final_project_2022/parquet_airlines_data/")

# COMMAND ----------

# Step 1: Total number of rows in the dataset
total_rows = df_flights_complete.count()
print(f"Total rows in dataset: {total_rows}")

# Step 2: Count number of nulls in DEP_DEL15 column
null_count = df_flights_complete.filter(col("DEP_DEL15").isNull()).count()
print(f"Total null values in DEP_DEL15: {null_count}")

# Step 3: Drop null values in DEP_DEL15 column
df_flights_filtered_complete = df_flights_complete.filter(col("DEP_DEL15").isNotNull())

# Step 4: Compute total flights and total delays per airline
df_delays_complete = (
    df_flights_filtered_complete
    .groupBy("OP_UNIQUE_CARRIER")
    .agg(
        count("*").alias("Total_Flights"),  # Count total flights per airline
        F.sum(col("DEP_DEL15")).alias("Total_Delayed_Flights")  # Count delays
    )
    .withColumn("Delay_Percentage", round((col("Total_Delayed_Flights") / col("Total_Flights")) * 100, 2))  # Calculate delay percentage
    .orderBy(desc("Delay_Percentage"))  # Sort by worst performer
)

# Show the top 10 airlines with the highest delay percentage
display(df_delays_complete.limit(10))

# COMMAND ----------

# Step 4: Compute total flights and total delays per airline
df_delays_complete = (
    df_flights_filtered_complete
    .groupBy("OP_UNIQUE_CARRIER")
    .agg(
        count("*").alias("Total_Flights"),  # Count total flights per airline
        F.sum(col("DEP_DEL15")).alias("Total_Delayed_Flights")  # Count delays
    )
    .withColumn("Delay_Percentage", round((col("Total_Delayed_Flights") / col("Total_Flights")) * 100, 2))  # Calculate delay percentage
    .orderBy(asc("Delay_Percentage"))  # Sort by best performer
)

# Show the top 10 airlines with the highest delay percentage
display(df_delays_complete.limit(10))

# COMMAND ----------

# Read Full Airlines Dataset (2015-2021)
df_flights_WN = spark.read.parquet(f"dbfs:/student-groups/Group_04_04/df_joined_2015_2024.parquet/")

# Filter for Southwest Airlines (WN) only
df_flights_WN = df_flights_WN.filter(col("OP_UNIQUE_CARRIER") == "WN")

# Extract year from flight date
df_flights_WN = df_flights_WN.withColumn("Year", year(col("FL_DATE")))

# Remove rows where DEP_DEL15 is null
df_flights_WN_filtered = df_flights_WN.filter(col("DEP_DEL15").isNotNull())

# Aggregate total flights and total delays per year
df_WN_yearly_delays = (
    df_flights_WN_filtered
    .groupBy("Year")
    .agg(
        count("*").alias("Total_Flights_WN"),
        F.sum(col("DEP_DEL15")).alias("Total_Delayed_Flights_WN")
    )
    .withColumn("Delay_Percentage_WN", (col("Total_Delayed_Flights_WN") / col("Total_Flights_WN")) * 100)  # Compute percentage
    .orderBy("Year")  # Order by year
)

# Show year-over-year delay statistics
df_WN_yearly_delays.show()

# COMMAND ----------

# Filter for years 2015-2024
df_WN_yearly_delays_filtered = df_WN_yearly_delays.filter((col("Year") >= 2015) & (col("Year") <= 2024))

# Convert Spark DataFrame to Pandas for visualization
df_WN_pd = df_WN_yearly_delays_filtered.toPandas()

# Handle NaN values by filling them with 0
df_WN_pd["Delay_Percentage_WN"].fillna(0, inplace=True)

# Compute mean delay percentage across all years
mean_delay_percentage = df_WN_pd["Delay_Percentage_WN"].mean()

# Create figure
plt.figure(figsize=(10, 6))

# Plot bar chart
bars = plt.bar(df_WN_pd["Year"].astype(str), df_WN_pd["Delay_Percentage_WN"], color="skyblue", label="Yearly Delay Percentage")

# Add a horizontal mean line
plt.axhline(y=mean_delay_percentage, color='red', linestyle='--', linewidth=1, label=f"Mean Delay Percentage ({mean_delay_percentage:.2f}%)")

# Add text labels slightly lower for better visibility
for bar in bars:
    yval = bar.get_height()  # Get height of bar (delay percentage)
    plt.text(bar.get_x() + bar.get_width()/2, yval - 1, f"{yval:.2f}%", ha='center', fontsize=10)

# Customize chart
plt.xlabel("Year")
plt.ylabel("Delay Percentage (%)")
plt.title("Southwest Airlines Year-over-Year Flight Delay Trend (2015-2024)")
plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
plt.legend()

# Remove the grey dotted lines
plt.grid(False)  # Disables the grid completely

# Save the plot as an image
chart_path = "/dbfs/FileStore/WN_flight_delays_2015_2024.png"  # Path in DBFS
plt.savefig(chart_path, dpi=300, bbox_inches='tight')  # High-quality PNG file

# Show plot
plt.show()

# Print path for reference
print(f"Chart saved at: {chart_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Query for Custom Join

# COMMAND ----------

"""Extract zipped weather data from 2022-2024"""

for year in range(2022, 2025):
    local_tar_path = f"/tmp/weather_{year}.tar.gz"
    local_extract_path = f"/tmp/extracted/weather_{year}"

    # Copy tar.gz from DBFS to local driver disk
    dbutils.fs.cp(f"dbfs:/FileStore/tables/weather_{year}.tar.gz", f"file:{local_tar_path}")

    # Extract tar.gz locally
    with tarfile.open(local_tar_path, "r:gz") as tar:
        tar.extractall(path=local_extract_path)

    # Copy extracted files back to DBFS
    for root, dirs, files in os.walk(local_extract_path):
        for file in files:
            local_file = os.path.join(root, file)
            # Construct DBFS destination path, e.g., keep folder structure if needed
            relative_path = os.path.relpath(local_file, local_extract_path)
            dbfs_path = f"dbfs:/FileStore/tables/extracted_weather/{relative_path}"
            dbutils.fs.cp(f"file:{local_file}", dbfs_path)

# COMMAND ----------

"""Read extracted weather data from 2022-2024 and checkpoint as parquet"""

df_weather_new = spark.read.csv(
    "dbfs:/FileStore/tables/extracted_weather/*.csv",
    header=True
)

rename_dict = {
    'MonthlyAverageWindSpeed': 'AWND',
    'MonthlyHeatingDegreeDays': 'HTDD',
    'HeatingDegreeDaysSeasonToDate': 'HDSD',
    'MonthlyNumberDaysWithSnowfall': 'DSNW',
    'CoolingDegreeDaysSeasonToDate': 'CDSD',
    'MonthlyCoolingDegreeDays': 'CLDD'
}

for old, new in rename_dict.items():
    df_weather_new = df_weather_new.withColumnRenamed(old, new)

df_weather_new = df_weather_new.drop("MonthlyNumberDaysWithThunderstorms", "MonthlyNumberDaysWithHeavyFog")
df_weather_new = df_weather_new.withColumn("YEAR", substring(col("DATE"), 1, 4).cast("int"))
df_weather_new.write.mode("overwrite").parquet(f"dbfs:/student-groups/Group_04_04/df_weather_2022_2024.parquet")

# COMMAND ----------

"""Function to join flights and weather"""

def join_flights_weather(df_flights, df_weather, df_airport_codes):
    df_flights.createOrReplaceTempView("flights")
    df_weather.createOrReplaceTempView("weather")
    df_airport_codes.createOrReplaceTempView("airport_codes")

    df_weather2 = spark.sql("""
        WITH weather_deduped AS (
            SELECT
                DISTINCT *
            FROM
                weather
        )
        SELECT
            * EXCEPT (DATE, YEAR),
            TIMESTAMP(
                make_timestamp(
                    CAST(SUBSTR(DATE, 1, 4) AS INT),
                    CAST(SUBSTR(DATE, 6, 2) AS INT),
                    CAST(SUBSTR(DATE, 9, 2) AS INT),
                    CAST(SUBSTR(DATE, 12, 2) AS INT),
                    CAST(SUBSTR(DATE, 15, 2) AS INT),
                    CAST(SUBSTR(DATE, 18, 2) AS INT)
                )
            ) AS DATE
        FROM
            weather_deduped
    """)

    df_flights2 = spark.sql("""
        WITH ac AS (
            SELECT
                iata_code,
                name,
                type,
                CAST(SPLIT_PART(coordinates, ',', 1) AS FLOAT) AS longitude,
                CAST(SPLIT_PART(coordinates, ',', 2) AS FLOAT) AS latitude
            FROM
                airport_codes
            WHERE
                iata_code IS NOT NULL
        ),
        s AS (
            SELECT
                DISTINCT STATION as neighbor_id,
                NAME as neighbor_name,
                CAST(LATITUDE AS FLOAT) AS neighbor_lat,
                CAST(LONGITUDE AS FLOAT) AS neighbor_lon
            FROM
                weather
        ),
        distances AS (
            SELECT
                ac.iata_code,
                ac.name,
                ac.type,
                ac.latitude,
                ac.longitude,
                s.neighbor_id,
                s.neighbor_name,
                s.neighbor_lat,
                s.neighbor_lon,
                6391 * 2 * ASIN(SQRT(POWER(SIN(RADIANS(s.neighbor_lat - ac.latitude) / 2), 2) + COS(RADIANS(ac.latitude)) * COS(RADIANS(s.neighbor_lat)) * POWER(SIN(RADIANS(s.neighbor_lon - ac.longitude) / 2), 2))) AS distance_km
            FROM
                ac CROSS JOIN s
        ),
        distance_rankings AS (
            SELECT
                *,
                ROW_NUMBER() OVER (PARTITION BY iata_code ORDER BY distance_km ASC) AS distance_rank
            FROM
                distances
        ),
        nearest_neighbor AS (
            SELECT
                iata_code,
                name AS airport_name,
                latitude AS airport_lat,
                longitude AS airport_lon,
                type,
                neighbor_id,
                neighbor_name,
                distance_km
            FROM
                distance_rankings
            WHERE
                distance_rank = 1
            ORDER BY
                distance_km ASC
        ),
        state_time_zones AS (
            SELECT 
                state, 
                time_zone_offset
            FROM 
                (
                    VALUES
                        ('AL', 'America/Chicago'), 
                        ('AK', 'America/Anchorage'), 
                        ('AZ', 'America/Phoenix'), 
                        ('AR', 'America/Chicago'), 
                        ('CA', 'America/Los_Angeles'), 
                        ('CO', 'America/Denver'), 
                        ('CT', 'America/New_York'), 
                        ('DE', 'America/New_York'), 
                        ('FL', 'America/New_York'), 
                        ('GA', 'America/New_York'), 
                        ('HI', 'Pacific/Honolulu'), 
                        ('ID', 'America/Boise'), 
                        ('IL', 'America/Chicago'), 
                        ('IN', 'America/New_York'), 
                        ('IA', 'America/Chicago'), 
                        ('KS', 'America/Chicago'), 
                        ('KY', 'America/New_York'), 
                        ('LA', 'America/Chicago'), 
                        ('ME', 'America/New_York'), 
                        ('MD', 'America/New_York'), 
                        ('MA', 'America/New_York'), 
                        ('MI', 'America/New_York'), 
                        ('MN', 'America/Chicago'), 
                        ('MS', 'America/Chicago'), 
                        ('MO', 'America/Chicago'), 
                        ('MT', 'America/Denver'), 
                        ('NE', 'America/Chicago'), 
                        ('NV', 'America/Los_Angeles'), 
                        ('NH', 'America/New_York'), 
                        ('NJ', 'America/New_York'), 
                        ('NM', 'America/Denver'), 
                        ('NY', 'America/New_York'), 
                        ('NC', 'America/New_York'), 
                        ('ND', 'America/Chicago'), 
                        ('OH', 'America/New_York'), 
                        ('OK', 'America/Chicago'), 
                        ('OR', 'America/Los_Angeles'), 
                        ('PA', 'America/New_York'), 
                        ('PR', 'America/Puerto_Rico'),
                        ('RI', 'America/New_York'), 
                        ('SC', 'America/New_York'), 
                        ('SD', 'America/Chicago'), 
                        ('TN', 'America/Chicago'), 
                        ('TX', 'America/Chicago'), 
                        ('UT', 'America/Denver'), 
                        ('VT', 'America/New_York'), 
                        ('VA', 'America/New_York'), 
                        ('WA', 'America/Los_Angeles'), 
                        ('WV', 'America/New_York'), 
                        ('WI', 'America/Chicago'), 
                        ('WY', 'America/Denver')
                ) AS t(state, time_zone_offset)
        ),
        flights_deduped AS (
            SELECT
                DISTINCT * EXCEPT (FL_DATE),
                SUBSTRING(FL_DATE, 1, 10) AS FL_DATE
            FROM
                flights
        ),
        flights_with_stations AS (
            SELECT
                f.*,
                TO_TIMESTAMP(
                    CONCAT(
                        FL_DATE,
                        ' ', 
                        CASE 
                            WHEN LENGTH(CRS_DEP_TIME) = 3 THEN CONCAT('0', SUBSTR(CRS_DEP_TIME, 1, 1), ':', SUBSTR(CRS_DEP_TIME, 2, 2))
                            ELSE CONCAT(SUBSTR(CRS_DEP_TIME, 1, 2), ':', SUBSTR(CRS_DEP_TIME, 3, 2))
                        END,
                        ':00'
                    ),
                    'yyyy-MM-dd HH:mm:ss'
                ) AS sched_depart_date_time,
                to_utc_timestamp(sched_depart_date_time, st.time_zone_offset) AS sched_depart_date_time_UTC,
                TO_TIMESTAMP(
                    CONCAT(
                        FL_DATE,
                        ' ', 
                        CASE 
                            WHEN LENGTH(DEP_TIME) = 3 THEN CONCAT('0', SUBSTR(DEP_TIME, 1, 1), ':', SUBSTR(DEP_TIME, 2, 2))
                            ELSE CONCAT(SUBSTR(DEP_TIME, 1, 2), ':', SUBSTR(DEP_TIME, 3, 2))
                        END,
                        ':00'
                    ),
                    'yyyy-MM-dd HH:mm:ss'
                ) AS depart_date_time,
                to_utc_timestamp(depart_date_time, st.time_zone_offset) AS depart_date_time_UTC,
                TIMESTAMPADD(MINUTE, ACTUAL_ELAPSED_TIME, depart_date_time_UTC) AS arrival_date_time_UTC,
                sched_depart_date_time_UTC - INTERVAL 4 HOUR AS four_hours_prior_depart_UTC,
                sched_depart_date_time_UTC - INTERVAL 2 HOUR AS two_hours_prior_depart_UTC,
                nn.airport_name AS origin_airport_name,
                nn.type AS origin_type,
                nn.airport_lat AS origin_airport_lat,
                nn.airport_lon AS origin_airport_lon,
                nn.neighbor_id AS origin_station_id,
                nn.neighbor_name AS origin_station_name,
                nn.distance_km AS origin_dis,
                nn2.airport_name AS dest_airport_name,
                nn2.type AS dest_type,
                nn2.airport_lat AS dest_airport_lat,
                nn2.airport_lon AS dest_airport_lon,
                nn2.neighbor_id AS dest_station_id,
                nn2.neighbor_name AS dest_station_name,
                nn2.distance_km AS dest_dis
            FROM
                flights_deduped f
                LEFT JOIN state_time_zones st
                    ON f.ORIGIN_STATE_ABR = st.state
                LEFT JOIN nearest_neighbor nn
                    ON f.ORIGIN = nn.iata_code
                LEFT JOIN nearest_neighbor nn2
                    ON f.DEST = nn2.iata_code
            ORDER BY
                TAIL_NUM DESC,
                sched_depart_date_time_UTC
        )
        SELECT
            *,
            CASE
                WHEN TAIL_NUM IS NOT NULL THEN LAG(CANCELLED) OVER (PARTITION BY TAIL_NUM ORDER BY sched_depart_date_time_UTC)
                ELSE NULL
            END AS PREV_CANCELLED,
            CASE
                WHEN TAIL_NUM IS NOT NULL THEN LAG(ORIGIN) OVER (PARTITION BY TAIL_NUM ORDER BY sched_depart_date_time_UTC)
                ELSE NULL
            END AS PREV_ORIGIN,
            CASE
                WHEN TAIL_NUM IS NOT NULL THEN LAG(arrival_date_time_UTC) OVER (PARTITION BY TAIL_NUM ORDER BY sched_depart_date_time_UTC)
                ELSE NULL
            END AS PREV_arrival_date_time_UTC,
            TIMESTAMPDIFF(MINUTE, PREV_arrival_date_time_UTC, sched_depart_date_time_UTC) AS MINUTES_BETWEEN_FLIGHTS,
            CASE
                WHEN TAIL_NUM IS NOT NULL THEN LAG(ARR_DELAY) OVER (PARTITION BY TAIL_NUM ORDER BY sched_depart_date_time_UTC)
                ELSE NULL
            END AS PREV_ARR_DELAY,
            CASE
                WHEN TAIL_NUM IS NOT NULL THEN LAG(ARR_DELAY_NEW) OVER (PARTITION BY TAIL_NUM ORDER BY sched_depart_date_time_UTC)
                ELSE NULL
            END AS PREV_ARR_DELAY_NEW,
            CASE
                WHEN TAIL_NUM IS NOT NULL THEN LAG(ARR_DEL15) OVER (PARTITION BY TAIL_NUM ORDER BY sched_depart_date_time_UTC)
                ELSE NULL
            END AS PREV_ARR_DEL15,
            CASE
                WHEN TAIL_NUM IS NOT NULL THEN CONCAT_WS('->', LAG(ORIGIN) OVER (PARTITION BY TAIL_NUM ORDER BY sched_depart_date_time_UTC), ORIGIN, DEST)
                ELSE NULL
            END AS TRIPLET
        FROM
            flights_with_stations
    """)

    df_flights2 = df_flights2.withColumn("join_date", F.date_trunc("hour", F.col("four_hours_prior_depart_UTC")))
    df_weather2 = df_weather2.withColumn("join_date", F.date_trunc("hour", F.col("DATE")))

    df_joined = (
        df_flights2
        .join(
            df_weather2,
            (
                (df_flights2.origin_station_id == df_weather2.STATION) &
                (df_flights2.join_date == df_weather2.join_date) &
                (df_weather2.DATE.between(
                    df_flights2.four_hours_prior_depart_UTC,
                    df_flights2.two_hours_prior_depart_UTC
                ))
            ),
            how='left'
        )
        .drop(df_weather2.join_date)
    )

    return df_joined


# COMMAND ----------

"""Read flights data from 2022-2024 and checkpoint as parquet"""

file_paths = [
    f"dbfs:/FileStore/tables/flights_{year}_{month:02d}.csv"
    for year in range(2022, 2025)
    for month in range(1, 13)
]

df_flights_new = spark.read.option("header", True).option("inferSchema", True).csv(file_paths)

# List of columns to cast to StringType
columns_to_cast = [
    'DIV3_TOTAL_GTIME', 'DIV3_LONGEST_GTIME', 'DIV2_LONGEST_GTIME',
    'DIV3_AIRPORT_ID', 'DIV3_AIRPORT_SEQ_ID', 'DIV2_AIRPORT_ID',
    'DIV2_TOTAL_GTIME', 'DIV3_WHEELS_OFF', 'DIV3_WHEELS_ON',
    'DIV2_WHEELS_OFF', 'DIV2_WHEELS_ON', 'DIV2_AIRPORT_SEQ_ID'
]

# Cast all specified columns to StringType
for column in columns_to_cast:
    df_flights_new = df_flights_new.withColumn(column, col(column).cast("string"))

df_flights_new = df_flights_new.withColumn(
    "FL_DATE",
    date_format(
        to_timestamp(col("FL_DATE"), "M/d/yyyy h:mm:ss a"),
        "yyyy-MM-dd"
    )
)

df_flights_new.write.mode("overwrite").parquet(f"dbfs:/student-groups/Group_04_04/df_flights_2022_2024.parquet")

# COMMAND ----------

"""Join and union old and new flights and weather data"""

df_flights_old = spark.read.parquet(f"dbfs:/mnt/mids-w261/datasets_final_project_2022/parquet_airlines_data")
df_weather_old = spark.read.parquet(f"dbfs:/mnt/mids-w261/datasets_final_project_2022/parquet_weather_data")
df_flights_new = spark.read.parquet(f"dbfs:/student-groups/Group_04_04/df_flights_2022_2024.parquet")
df_weather_new = spark.read.parquet(f"dbfs:/student-groups/Group_04_04/df_weather_2022_2024.parquet")
df_airport_codes = spark.read.format("csv").option("header","true").load(f"dbfs:/mnt/mids-w261/airport-codes_csv.csv")

df_joined_old = join_flights_weather(df_flights_old, df_weather_old, df_airport_codes)
df_joined_new = join_flights_weather(df_flights_new, df_weather_new, df_airport_codes)
df_joined = df_joined_old.unionByName(df_joined_new)

df_joined.write.mode("overwrite").parquet(f"dbfs:/student-groups/Group_04_04/df_joined_2015_2024.parquet")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Exploratory Data Analysis - Full Data

# COMMAND ----------

# Save the combined DataFrame to a temporary parquet file to check its size
temp_parquet_path = "dbfs:/student-groups/Group_04_04/df_joined_2015_2024.parquet"
# df_otpw_combined_60M.write.mode("overwrite").parquet(temp_parquet_path)

# Get the size of the parquet file
file_info = dbutils.fs.ls(temp_parquet_path)
total_size = sum(file.size for file in file_info)
print(f"Total size of the combined DataFrame: {total_size / (1024 * 1024 * 1024):.2f} GB")

# COMMAND ----------

folder_path = f"dbfs:/student-groups/Group_04_04"

# Save df_southwest_cleaned as a parquet file
df_otpw_combined_60M.write.parquet(f"{folder_path}/df_otpw_60M_complete.parquet") # use to save a new version
# df_southwest_cleaned.write.mode('overwrite').parquet(f"{folder_path}/df_otpw_12M_southwest_cleaned_updated_Apr_02.parquet") # use if you want to overwrite exisiting file

# COMMAND ----------

data_BASE_DIR = "dbfs:/student-groups/Group_04_04/"
# display(dbutils.fs.ls(f"{data_BASE_DIR}"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Reading and Processing Data from the Custom Join

# COMMAND ----------

# df_otpw_60M_complete_parquet = spark.read.parquet(f"dbfs:/student-groups/Group_04_04/df_joined_full_with_lag.parquet/")
df_otpw_60M_complete_parquet = spark.read.parquet(f"dbfs:/student-groups/Group_04_04/df_joined_2015_2024.parquet/")

# COMMAND ----------

display(df_otpw_60M_complete_parquet)

# COMMAND ----------

# Compute null percentage per column
def calculate_null_percentage(df):
    # Count total rows for reference
    total_rows = df.count()
    
    null_percentage_df = df.select([
        (count(when(col(c).isNull() | (col(c) == ""), c)) / total_rows * 100).alias(c)
        for c in df.columns
    ])
    return null_percentage_df

# COMMAND ----------

# ----------------------------- #
# ✅ Check Dataset Shape (Rows & Columns)
# ----------------------------- #
num_rows = df_otpw_60M_complete_parquet.count()
num_cols = len(df_otpw_60M_complete_parquet.columns)

print(f"✅ Dataset contains {num_rows:,} rows and {num_cols} columns.")

null_percentage_df = calculate_null_percentage(df_otpw_60M_complete_parquet)
display(null_percentage_df)

# COMMAND ----------

distinct_years = df_otpw_60M_complete_parquet.select("YEAR").distinct()
display(distinct_years.orderBy("YEAR", ascending=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Drop unwanted cols

# COMMAND ----------

delay_columns_to_remove = [
    "DEP_DELAY",           # Actual departure delay in minutes
    "DEP_DELAY_NEW",       # Departure delay, with negative values set to 0
    "DEP_DELAY_GROUP",     # Departure delay groups
    "ARR_DELAY",           # Arrival delay in minutes
    "ARR_DELAY_NEW",       # Arrival delay, with negative values set to 0
    "ARR_DEL15",           # Arrival delay indicator (15+ minutes)
    "ARR_DELAY_GROUP",     # Arrival delay groups
]

# COMMAND ----------

actual_time_columns_to_remove = [
    "DEP_TIME",            # Actual departure time
    "TAXI_OUT",            # Taxi out time
    "WHEELS_OFF",          # Wheels off time
    "WHEELS_ON",           # Wheels on time
    "TAXI_IN",             # Taxi in time
    "ARR_TIME",            # Actual arrival time
    "ACTUAL_ELAPSED_TIME", # Actual elapsed time
    "AIR_TIME"             # Air time
]

# COMMAND ----------

# Create a list to store column names with null percentage <= 50%
columns_with_low_nulls = []

# Get the threshold
threshold = 50.0

# Collect the single row of percentages
null_percentages = null_percentage_df.collect()[0]

# Get the column names that meet the threshold
columns_with_low_nulls = [col_name for col_name in null_percentage_df.columns 
                         if null_percentages[col_name] <= threshold]

# Additional columns to remove
columns_to_remove = delay_columns_to_remove + actual_time_columns_to_remove

# Filter out these columns from your list of columns with low nulls
filtered_columns = [col for col in columns_with_low_nulls if col not in columns_to_remove]

# Print the results
print(f"Found {len(columns_with_low_nulls)} columns with null percentage <= 50%")
print(f"Found {len(columns_to_remove)} columns that could cause data leakage")
print(f"Found {len(filtered_columns)} columns to be used finally")

# Create a new DataFrame with only these columns if needed
df_otpw_60M_complete_parquet_col_filtered = df_otpw_60M_complete_parquet.select(filtered_columns)

# COMMAND ----------

columns_with_low_nulls

# COMMAND ----------

display(df_otpw_60M_complete_parquet_col_filtered)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Basic cleaning of selected data

# COMMAND ----------

def convert_column_types(df):
    """
    Function to convert column data types appropriately for the flight delay dataset,
    after removing columns that could cause data leakage.
    
    Args:
        df: PySpark DataFrame to convert
        
    Returns:
        PySpark DataFrame with converted column types
    """
    # Define column type mappings
    
    # Date columns - date fields
    date_columns = ["FL_DATE", "DATE"]
    
    # Timestamp columns - date-time fields with time zone information
    timestamp_columns = [
        "sched_depart_date_time_UTC", "four_hours_prior_depart_UTC", 
        "two_hours_prior_depart_UTC"
    ]
    
    # Integer columns - IDs, counts, whole numbers
    integer_columns = [
        "YEAR", "QUARTER", "MONTH", "DAY_OF_MONTH", "DAY_OF_WEEK",
        "OP_CARRIER_AIRLINE_ID", "OP_CARRIER_FL_NUM", 
        "ORIGIN_AIRPORT_ID", "ORIGIN_AIRPORT_SEQ_ID", "ORIGIN_CITY_MARKET_ID",
        "ORIGIN_STATE_FIPS", "ORIGIN_WAC",
        "DEST_AIRPORT_ID", "DEST_AIRPORT_SEQ_ID", "DEST_CITY_MARKET_ID",
        "DEST_STATE_FIPS", "DEST_WAC",
        "CRS_DEP_TIME", "CRS_ARR_TIME", "FLIGHTS", "DISTANCE_GROUP"
    ]
    
    # Double columns - measurements, calculations with decimal points
    double_columns = [
        "DEP_DEL15",       # Keep target variable
        "CRS_ELAPSED_TIME", "DISTANCE",
        "CANCELLED", "DIVERTED",
        "origin_station_lat", "origin_station_lon", "origin_airport_lat", 
        "origin_airport_lon", "origin_station_dis",
        "dest_station_lat", "dest_station_lon", "dest_airport_lat", 
        "dest_airport_lon", "dest_station_dis",
        "LATITUDE", "LONGITUDE", "ELEVATION"
    ]
    
    # Decimal columns - weather measurements that need higher precision
    decimal_columns = [
        "HourlyAltimeterSetting", "HourlyDewPointTemperature", "HourlyDryBulbTemperature", 
        "HourlyPrecipitation", "HourlyRelativeHumidity", "HourlySeaLevelPressure", 
        "HourlyStationPressure", "HourlyVisibility", "HourlyWetBulbTemperature", 
        "HourlyWindDirection", "HourlyWindSpeed"
    ]
    
    # String columns - text identifiers, codes, names, etc.
    string_columns = [
        "OP_UNIQUE_CARRIER", "OP_CARRIER", "TAIL_NUM", 
        "ORIGIN", "DEST", "ORIGIN_CITY_NAME", "DEST_CITY_NAME", 
        "ORIGIN_STATE_ABR", "DEST_STATE_ABR", "ORIGIN_STATE_NM", "DEST_STATE_NM", 
        "DEP_TIME_BLK", "ARR_TIME_BLK", "HourlySkyConditions", "REM",
        "origin_airport_name", "origin_station_name", "origin_station_id", 
        "origin_iata_code", "origin_icao", "origin_type", "origin_region",
        "dest_airport_name", "dest_station_name", "dest_station_id", 
        "dest_iata_code", "dest_icao", "dest_type", "dest_region",
        "STATION", "NAME", "REPORT_TYPE", "SOURCE", "WindEquipmentChangeDate", "TRIPLET"
    ]
    
    # Convert date columns
    for column in date_columns:
        if column in df.columns:
            df = df.withColumn(column, to_date(col(column)))
    
    # Convert timestamp columns
    for column in timestamp_columns:
        if column in df.columns:
            df = df.withColumn(column, to_timestamp(col(column)))
    
    # Convert integer columns
    for column in integer_columns:
        if column in df.columns:
            df = df.withColumn(column, col(column).cast(IntegerType()))
    
    # Convert double columns
    for column in double_columns:
        if column in df.columns:
            df = df.withColumn(column, col(column).cast(DoubleType()))
    
    # Convert decimal columns (with higher precision for weather data)
    for column in decimal_columns:
        if column in df.columns:
            df = df.withColumn(column, col(column).cast(DecimalType(10, 4)))
    
    # Convert string columns
    for column in string_columns:
        if column in df.columns:
            df = df.withColumn(column, col(column).cast(StringType()))
    
    return df

# COMMAND ----------

def clean_flight_data(df):
    """
    Comprehensive data cleaning function for airline dataset addressing all columns with null values.
    
    Args:
        df: PySpark DataFrame with flight and weather data
        
    Returns:
        Cleaned PySpark DataFrame with no null values
    """
    # 1. First sort the data to ensure proper time-based operations
    df = df.orderBy("ORIGIN", "FL_DATE", "CRS_DEP_TIME")
    
    # 2. Handle DEP_DEL15 for cancelled flights - assign value 2 to represent "cancelled"
    if "DEP_DEL15" in df.columns and "CANCELLED" in df.columns:
        df = df.withColumn(
            "DEP_DEL15", 
            F.when(F.col("DEP_DEL15").isNull() & (F.col("CANCELLED") == 1), 2).otherwise(F.col("DEP_DEL15"))
        )
    
    # 3. Drop rows with missing values in critical columns INCLUDING DEP_DEL15 after imputation
    critical_cols = ["FL_DATE", "OP_CARRIER", "ORIGIN", "DEST", "CRS_DEP_TIME", "CRS_ARR_TIME", "TAIL_NUM", "DEP_DEL15"]
    for col in critical_cols:
        if col in df.columns:
            df = df.filter(F.col(col).isNotNull())
    
    # 4. Calculate CRS_ELAPSED_TIME from CRS_DEP_TIME and CRS_ARR_TIME if missing
    if "CRS_ELAPSED_TIME" in df.columns and "CRS_DEP_TIME" in df.columns and "CRS_ARR_TIME" in df.columns:
        df = df.withColumn(
            "CRS_ELAPSED_TIME_CALC",
            F.when(
                F.col("CRS_ARR_TIME") < F.col("CRS_DEP_TIME"),  # Handle overnight flights
                ((F.col("CRS_ARR_TIME") + 2400) - F.col("CRS_DEP_TIME"))
            ).otherwise(
                F.col("CRS_ARR_TIME") - F.col("CRS_DEP_TIME")
            )
        )
        
        # Convert from HHMM format to minutes
        df = df.withColumn(
            "CRS_ELAPSED_TIME_CALC",
            (F.floor(F.col("CRS_ELAPSED_TIME_CALC")/F.lit(100)) * F.lit(60)) + 
            (F.col("CRS_ELAPSED_TIME_CALC") % F.lit(100))
        )
        
        df = df.withColumn("CRS_ELAPSED_TIME", 
                          F.coalesce(F.col("CRS_ELAPSED_TIME"), F.col("CRS_ELAPSED_TIME_CALC")))
        df = df.drop("CRS_ELAPSED_TIME_CALC")
    
    # 5. Extract temporal features from FL_DATE if missing
    date_derived_cols = {
        "YEAR": F.year("FL_DATE"),
        "QUARTER": F.quarter("FL_DATE"),
        "MONTH": F.month("FL_DATE"),
        "DAY_OF_WEEK": F.dayofweek("FL_DATE"),
        "DAY_OF_MONTH": F.dayofmonth("FL_DATE")
    }
    
    for col_name, expr in date_derived_cols.items():
        if col_name in df.columns:
            df = df.withColumn(col_name, F.coalesce(F.col(col_name), expr))
    
    # 6. Handle weather data with null values - using BACKWARD-LOOKING windows only to prevent data leakage
    # Include ALL weather columns that have nulls
    weather_cols = [
        "HourlyWindDirection", "HourlyAltimeterSetting", "HourlySkyConditions",
        "HourlyVisibility", "HourlyDewPointTemperature", "HourlyWindSpeed",
        "HourlyDryBulbTemperature", "HourlyPrecipitation", "HourlyRelativeHumidity",
        "HourlySeaLevelPressure", "HourlyStationPressure", "HourlyWetBulbTemperature"
    ]
    
    # Use a backward-looking window to avoid data leakage
    if any(col in df.columns for col in weather_cols):
        # Create window specs once - only looking at past values (negative range)
        # Increase window size to capture more historical data
        origin_date_window = Window.partitionBy("ORIGIN", F.to_date("FL_DATE")).orderBy("CRS_DEP_TIME").rowsBetween(-10, 0)
        origin_past_window = Window.partitionBy("ORIGIN").orderBy("FL_DATE", "CRS_DEP_TIME").rowsBetween(-20, 0)
        origin_month_window = Window.partitionBy("ORIGIN", "MONTH")
        
        # Process numeric weather columns with backward-looking average
        numeric_weather_cols = [
            "HourlyWindDirection", "HourlyAltimeterSetting", "HourlyVisibility", 
            "HourlyDewPointTemperature", "HourlyWindSpeed", "HourlyDryBulbTemperature",
            "HourlyPrecipitation", "HourlyRelativeHumidity", "HourlySeaLevelPressure", 
            "HourlyStationPressure", "HourlyWetBulbTemperature"
        ]
        
        for col in numeric_weather_cols:
            if col in df.columns:
                # First try rolling average within same day/origin (past values only)
                df = df.withColumn(f"{col}_rolling", F.avg(F.col(col)).over(origin_date_window))
                
                # Then try past days at same origin if still null
                df = df.withColumn(f"{col}_past", F.avg(F.col(col)).over(origin_past_window))
                
                # Then try origin/month median if still null
                df = df.withColumn(f"{col}_median", F.expr(f"percentile_approx({col}, 0.5)").over(origin_month_window))
                
                # Apply all imputations in one operation
                df = df.withColumn(col, F.coalesce(
                    F.col(col),
                    F.col(f"{col}_rolling"),
                    F.col(f"{col}_past"),
                    F.col(f"{col}_median")
                ))
                
                # Drop temporary columns
                df = df.drop(f"{col}_rolling", f"{col}_past", f"{col}_median")
                
                # Final global median fallback for any remaining nulls
                median_val = df.filter(F.col(col).isNotNull()).select(
                    F.expr(f"percentile_approx({col}, 0.5)").alias("median")
                ).first()["median"]
                
                if median_val is not None:
                    df = df.withColumn(col, F.coalesce(F.col(col), F.lit(median_val)))
                else:
                    # If even the global median is null, use a reasonable default based on column type
                    if "Temperature" in col:
                        df = df.withColumn(col, F.coalesce(F.col(col), F.lit(0.0)))  # Default temperature
                    elif "Direction" in col:
                        df = df.withColumn(col, F.coalesce(F.col(col), F.lit(0.0)))  # Default direction
                    elif "Speed" in col:
                        df = df.withColumn(col, F.coalesce(F.col(col), F.lit(0.0)))  # Default speed
                    elif "Precipitation" in col:
                        df = df.withColumn(col, F.coalesce(F.col(col), F.lit(0.0)))  # Default precipitation
                    elif "Humidity" in col:
                        df = df.withColumn(col, F.coalesce(F.col(col), F.lit(50.0)))  # Default humidity
                    elif "Pressure" in col:
                        df = df.withColumn(col, F.coalesce(F.col(col), F.lit(1013.25)))  # Default pressure (standard atmosphere)
                    elif "Visibility" in col:
                        df = df.withColumn(col, F.coalesce(F.col(col), F.lit(10.0)))  # Default visibility
                    else:
                        df = df.withColumn(col, F.coalesce(F.col(col), F.lit(0.0)))  # Generic default
        
        # Handle HourlySkyConditions separately (categorical)
        if "HourlySkyConditions" in df.columns:
            # First try mode within same day/origin
            df = df.withColumn("sky_mode_day", 
                              F.first(F.col("HourlySkyConditions"), ignorenulls=True).over(origin_date_window))
            
            # Then try mode for origin/month
            df = df.withColumn("sky_mode_month", 
                              F.first(F.col("HourlySkyConditions"), ignorenulls=True).over(origin_month_window))
            
            # Apply imputations
            df = df.withColumn("HourlySkyConditions", F.coalesce(
                F.col("HourlySkyConditions"), 
                F.col("sky_mode_day"),
                F.col("sky_mode_month")
            ))
            
            # Drop temporary columns
            df = df.drop("sky_mode_day", "sky_mode_month")
            
            # Final global mode fallback
            mode_val = df.filter(F.col("HourlySkyConditions").isNotNull()).groupBy("HourlySkyConditions").count() \
                         .orderBy(F.desc("count")).limit(1).select("HourlySkyConditions").first()
            
            if mode_val is not None:
                mode_val = mode_val["HourlySkyConditions"]
                df = df.withColumn("HourlySkyConditions", F.coalesce(F.col("HourlySkyConditions"), F.lit(mode_val)))
            else:
                # Default sky condition if no mode can be determined
                df = df.withColumn("HourlySkyConditions", F.coalesce(F.col("HourlySkyConditions"), F.lit("CLR")))
    
    # 7. Handle DEP_TIME_BLK with nulls
    if "DEP_TIME_BLK" in df.columns and "CRS_DEP_TIME" in df.columns:
        df = df.withColumn(
            "DEP_TIME_BLK_DERIVED", 
            F.concat(
                F.lpad(F.floor(F.col("CRS_DEP_TIME")/F.lit(100)), 2, "0"),
                F.lit("00-"),
                F.lpad(F.floor(F.col("CRS_DEP_TIME")/F.lit(100)) + F.lit(1), 2, "0"),
                F.lit("59")
            )
        )
        df = df.withColumn("DEP_TIME_BLK", F.coalesce(F.col("DEP_TIME_BLK"), F.col("DEP_TIME_BLK_DERIVED")))
        df = df.drop("DEP_TIME_BLK_DERIVED")
    
    # 8. Handle ARR_TIME_BLK with potential nulls
    if "ARR_TIME_BLK" in df.columns and "CRS_ARR_TIME" in df.columns:
        df = df.withColumn(
            "ARR_TIME_BLK_DERIVED", 
            F.concat(
                F.lpad(F.floor(F.col("CRS_ARR_TIME")/F.lit(100)), 2, "0"),
                F.lit("00-"),
                F.lpad(F.floor(F.col("CRS_ARR_TIME")/F.lit(100)) + F.lit(1), 2, "0"),
                F.lit("59")
            )
        )
        df = df.withColumn("ARR_TIME_BLK", F.coalesce(F.col("ARR_TIME_BLK"), F.col("ARR_TIME_BLK_DERIVED")))
        df = df.drop("ARR_TIME_BLK_DERIVED")
    
    # 9. Handle timestamp columns
    timestamp_cols = ["sched_depart_date_time_UTC", "four_hours_prior_depart_UTC", "two_hours_prior_depart_UTC"]
    
    if any(col in df.columns for col in timestamp_cols) and "FL_DATE" in df.columns and "CRS_DEP_TIME" in df.columns:
        # Create base timestamp from FL_DATE and CRS_DEP_TIME
        df = df.withColumn(
            "base_timestamp", 
            F.to_timestamp(
                F.concat(
                    F.date_format("FL_DATE", "yyyy-MM-dd"), 
                    F.lit(" "), 
                    F.lpad(F.floor(F.col("CRS_DEP_TIME")/F.lit(100)), 2, "0"),
                    F.lit(":"),
                    F.lpad(F.col("CRS_DEP_TIME") % F.lit(100), 2, "0"),
                    F.lit(":00")
                )
            )
        )
        
        # Apply to each timestamp column
        if "sched_depart_date_time_UTC" in df.columns:
            df = df.withColumn("sched_depart_date_time_UTC", 
                              F.coalesce(F.col("sched_depart_date_time_UTC"), F.col("base_timestamp")))
        
        if "four_hours_prior_depart_UTC" in df.columns:
            # Use unix_timestamp for arithmetic on timestamps (4 hours = 14400 seconds)
            df = df.withColumn("four_hours_prior_depart_UTC", 
                              F.coalesce(
                                  F.col("four_hours_prior_depart_UTC"), 
                                  F.from_unixtime(F.unix_timestamp(F.col("base_timestamp")) - F.lit(14400))
                              ))
        
        if "two_hours_prior_depart_UTC" in df.columns:
            # Use unix_timestamp for arithmetic on timestamps (2 hours = 7200 seconds)
            df = df.withColumn("two_hours_prior_depart_UTC", 
                              F.coalesce(
                                  F.col("two_hours_prior_depart_UTC"), 
                                  F.from_unixtime(F.unix_timestamp(F.col("base_timestamp")) - F.lit(7200))
                              ))
        
        df = df.drop("base_timestamp")
    
    # 10. Handle REM column (0.02% nulls)
    if "REM" in df.columns:
        df = df.withColumn("REM", F.coalesce(F.col("REM"), F.lit("UNKNOWN")))
    
    # 11. Handle WindEquipmentChangeDate (28.5% nulls)
    if "WindEquipmentChangeDate" in df.columns:
        # First try to use the most common value per station
        if "STATION" in df.columns:
            station_window = Window.partitionBy("STATION")
            df = df.withColumn("wind_equip_mode", 
                              F.first(F.col("WindEquipmentChangeDate"), ignorenulls=True).over(station_window))
            df = df.withColumn("WindEquipmentChangeDate", 
                              F.coalesce(F.col("WindEquipmentChangeDate"), F.col("wind_equip_mode")))
            df = df.drop("wind_equip_mode")
        
        # Then use a default value for any remaining nulls
        df = df.withColumn("WindEquipmentChangeDate", 
                          F.coalesce(F.col("WindEquipmentChangeDate"), F.lit("UNKNOWN")))
    
    # 12. Final check for any remaining nulls in any column
    for column in df.columns:
        # Check if column has any nulls
        null_count = df.filter(F.col(column).isNull()).count()
        
        if null_count > 0:
            # For any remaining nulls, use appropriate default values based on column type
            col_type = df.schema[column].dataType.simpleString()
            
            if "int" in col_type:
                df = df.withColumn(column, F.coalesce(F.col(column), F.lit(0)))
            elif "double" in col_type or "decimal" in col_type:
                df = df.withColumn(column, F.coalesce(F.col(column), F.lit(0.0)))
            elif "string" in col_type:
                df = df.withColumn(column, F.coalesce(F.col(column), F.lit("UNKNOWN")))
            elif "timestamp" in col_type:
                # Use FL_DATE as a base for any timestamp
                df = df.withColumn(column, 
                                  F.coalesce(F.col(column), F.to_timestamp(F.col("FL_DATE"))))
            elif "date" in col_type:
                df = df.withColumn(column, F.coalesce(F.col(column), F.col("FL_DATE")))
            else:
                # For any other types, use null (should not happen after all the above)
                pass
    
    return df

# COMMAND ----------

df_otpw_60M_complete_parquet_col_filtered_type_conversion = convert_column_types(df_otpw_60M_complete_parquet_col_filtered)

# COMMAND ----------

df_otpw_60M_complete_parquet_col_filtered_cleaned = clean_flight_data(df_otpw_60M_complete_parquet_col_filtered_type_conversion)

# COMMAND ----------

# Analyze missing values in cleaned features
null_percentage_df = calculate_null_percentage(df_otpw_60M_complete_parquet_col_filtered_cleaned)
display(null_percentage_df)

# COMMAND ----------

# display(df_otpw_60M_complete_parquet_col_filtered_cleaned)

# COMMAND ----------

folder_path = f"dbfs:/student-groups/Group_04_04"

# Save df_southwest_cleaned as a parquet file
df_otpw_60M_complete_parquet_col_filtered_cleaned.write.parquet(f"{folder_path}/df_otpw_60M_complete_parquet_col_filtered_cleaned.parquet") # use to save a new version
# df_otpw_60M_complete_parquet_col_filtered_cleaned.write.mode('overwrite').parquet(f"{folder_path}/df_otpw_60M_complete_parquet_col_filtered_cleaned.parquet") # use if you want to overwrite exisiting file

# COMMAND ----------

data_BASE_DIR = "dbfs:/student-groups/Group_04_04/"
display(dbutils.fs.ls(f"{data_BASE_DIR}"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Read the Cleaned Data and Start Data Engineering and Feature Selection

# COMMAND ----------

# df_otpw_60M_complete_parquet_cleaned = spark.read.parquet(f"dbfs:/student-groups/Group_04_04/df_otpw_60M_complete_parquet_col_filtered_cleaned.parquet/")
df_otpw_60M_complete_parquet_cleaned = df_otpw_60M_complete_parquet_col_filtered_cleaned

# COMMAND ----------

# MAGIC %md
# MAGIC ## Comprehensive Feature Engineering for Flight Delay Prediction
# MAGIC
# MAGIC ## Airport Profile Features
# MAGIC
# MAGIC | Feature | Description | Null Handling | Temporal Integrity |
# MAGIC |---------|-------------|---------------|---------------------|
# MAGIC | **Origin Airport Daily Operations** | Total number of flights departing from each origin airport on a given day | None needed (always populated) | Current day only |
# MAGIC | **Origin Airport 30-Day Rolling Volume** | Sum of flights from the origin airport over the past 30 days | 0 for first 30 days (no history available) | Growing window until 30 days history |
# MAGIC | **Origin Airport 1-Year Delay Rate** | Annual delay percentage at origin airport | Global fallback (15%) for first year (2015) rows | Expanding window using all prior data |
# MAGIC | **Route** | The origin and destination of the flight | Not needed | Concat the origin and destination in a single window |
# MAGIC | **Route Traffic Volume** | Number of flights between specific origin-destination pairs over the past year | 0 for new routes or first year (2015) rows | Expanding window using all prior data |
# MAGIC | **Southwest Market Share** | Percentage of flights operated by Southwest at each origin airport over the past year | 0 when no data available for Southwest flights | Rolling 365-day window |
# MAGIC | **Southwest Origin 30-Day Delay Rate** | Recent Southwest delay performance at origin airport (past 30 days) | Global fallback (15%) for missing data in the previous 30 days | Growing window until 30 days history |
# MAGIC | **Southwest Route Historical Performance** | Southwest's historical delay rate on specific routes over the past year | Global fallback (15%) for missing route data or first year rows | Expanding window using all prior data |
# MAGIC | **Southwest Relative Performance Index** | How Southwest compares to other airlines at the same airport (delay rate ratio) | Default value of 1.0 when no data available or division by zero occurs | Ratio with epsilon smoothing to prevent division by zero |
# MAGIC
# MAGIC
# MAGIC ## Time-Based Profile Features
# MAGIC
# MAGIC | Feature | Description | Null Handling | Temporal Integrity |
# MAGIC |---------|-------------|---------------|---------------------|
# MAGIC | **time_bucket** | 15-minute departure intervals | Derived from CRS_DEP_TIME (always populated) | Current flight only |
# MAGIC | **dep_hour** | Hour of day for scheduled departure | None needed | Current flight only |
# MAGIC | **time_of_day_category** | Morning/Midday/Evening/Night | Categorical fallback to "night" | Current flight only |
# MAGIC | **is_weekend** | Weekend flight indicator | None needed | Current flight only |
# MAGIC | **holiday_season** | Peak travel period indicator | None needed | Current flight only |
# MAGIC | **prior_day_delay_rate** | Previous day's delay rate at origin airport | 3-level fallback: prior day → airport avg → 15% global fallback | Strict date ordering |
# MAGIC | **same_day_prior_delay_percentage** | Percentage of flights delayed earlier in the day at the same airport | Additive smoothing (prevents 0/0) and nulls default to 0% delay rate | Same-day ordering |
# MAGIC | **time_based_congestion_ratio** | Current vs historical congestion ratio for the same time bucket (hour + 15-min interval) on the same day of the week at the same airport | 3-level fallback: historical average → airport avg → default capacity (10 flights) | 365-day lookback excluding current day |
# MAGIC
# MAGIC
# MAGIC ## Weather-Based Profile Features
# MAGIC
# MAGIC | Feature | Description | Calculation Method | Null Handling |
# MAGIC |---------|-------------|--------------------|---------------|
# MAGIC | **extreme_precipitation** | Flag for heavy precipitation | 95th percentile of historical precipitation data | 0 if missing |
# MAGIC | **extreme_wind** | Flag for high wind conditions | 95th percentile of historical wind speed data | 0 if missing |
# MAGIC | **extreme_temperature** | Flag for extreme temperatures | 5th/95th percentiles of historical temperature data | 0 if missing |
# MAGIC | **low_visibility** | Flag for poor visibility | 5th percentile of historical visibility data | 0 if missing |
# MAGIC | **extreme_weather_score** | Weighted weather risk score | Weighted sum of extreme conditions based on their historical delay impact | Scaled to [-1,1] |
# MAGIC | **heat_index** | Perceived temperature | NOAA heat index formula for T ≥ 80°F and RH ≥ 40% | Raw temp otherwise |
# MAGIC | **rapid_weather_change** | Significant weather shifts | Z-score > 3 in temp/wind over 24h window | 0 if missing data |
# MAGIC | **temp_anomaly_z** | Temperature deviation | Z-score vs. airport-month historical average | 0 if no history |
# MAGIC | **precip_anomaly_z** | Precipitation deviation | Z-score vs. airport-month historical average | 0 if no history |
# MAGIC
# MAGIC
# MAGIC ## Southwest Airlines Profile Features
# MAGIC
# MAGIC | Feature | Description | Calculation Method | Null Handling |
# MAGIC |---------|-------------|--------------------|---------------|
# MAGIC | **sw_time_of_day_delay_rate** | Southwest's delay rate by origin and time bucket | Expanding window average with origin/global fallbacks | Uses origin average → global median |
# MAGIC | **sw_day_of_week_delay_rate** | Bayesian-smoothed delay rate by route and weekday | (Delays + 3*global_p30)/(Flights + 3) | Built-in smoothing prevents nulls |
# MAGIC | **sw_aircraft_delay_rate** | Aircraft performance metric | Hierarchical: aircraft → route → global median | Always populated |
# MAGIC | **sw_origin_hub** | Dynamic hub identification | Top 15th percentile of Southwest flight volume | 0/1 encoding |
# MAGIC | **sw_schedule_buffer_ratio** | Schedule padding ratio | Current vs 1-year historical average | Defaults to 1.0 |
# MAGIC | **sw_origin_time_perf** | Hybrid airport/time performance | Time bucket → time category → global fallback | Hierarchical coalesce |
# MAGIC | **sw_route_importance** | Normalized route significance | (Flight count + distance) normalized | Always 0-2 range |

# COMMAND ----------

# MAGIC %md
# MAGIC #### Airport Profile Features

# COMMAND ----------

def add_airport_profile_features(df):
    """
    Robust airport profile features with temporal integrity and null handling
    """
    # Time basis for all rolling calculations
    # Time basis for all rolling calculations
    df = df.withColumn("days_since_epoch", F.datediff("FL_DATE", F.lit("1970-01-01")))
    
    # 1. Origin Airport Daily Operations
    window_daily = Window.partitionBy("ORIGIN", "FL_DATE")
    df = df.withColumn("daily_operations", F.count("*").over(window_daily))
    
    # 2. Origin Airport 30-Day Rolling Volume (Fixed window)
    window_30d = Window.partitionBy("ORIGIN") \
        .orderBy("days_since_epoch") \
        .rowsBetween(-29, 0)  # 30 rows (days)
    
    df = df.withColumn("rolling_30day_volume", 
                      F.coalesce(
                          F.sum("daily_operations").over(window_30d),
                          F.lit(0)
                      ))
    
    # 3. 1-Year Delay Rate (Fixed window)
    window_1yr = Window.partitionBy("ORIGIN") \
        .orderBy("days_since_epoch") \
        .rowsBetween(-365, 0)  # 365 rows (days)
    
    df = df.withColumn("origin_1yr_delay_rate", 
                      F.coalesce(
                          F.avg(F.col("DEP_DEL15")).over(window_1yr),
                          F.lit(0.15)  # Global average fallback
                      ))
    
    # 4. Route Traffic Volume (Expanding window)
    window_route = Window.partitionBy("ORIGIN", "DEST") \
        .orderBy("days_since_epoch") \
        .rowsBetween(Window.unboundedPreceding, 0)
    
    df = df.withColumn("route_1yr_volume", 
                      F.coalesce(
                          F.count("*").over(window_route),
                          F.lit(0)
                      ))
    
    # 5. Southwest Market Share (Fixed window)
    window_1yr_sw = Window.partitionBy("ORIGIN") \
        .orderBy("days_since_epoch") \
        .rowsBetween(-365, 0)  # 365 rows (days)
    
    df = df.withColumn("sw_market_share",
                      F.when(F.count("*").over(window_1yr_sw) > 0,
                            F.count(F.when(F.col("OP_CARRIER") == "WN", 1)).over(window_1yr_sw)
                            / F.count("*").over(window_1yr_sw))
                       .otherwise(0))
    
    # 6. Southwest Origin 30-Day Delay Rate (Same as window_30d)
    df = df.withColumn("sw_30d_delay",
                      F.coalesce(
                          F.avg(F.when(F.col("OP_CARRIER") == "WN", F.col("DEP_DEL15")))
                            .over(window_30d),
                          F.lit(0.15)
                      ))
    
    # 7. Southwest Route Performance (Expanding window)
    df = df.withColumn("sw_route_delay",
                      F.coalesce(
                          F.avg(F.when(F.col("OP_CARRIER") == "WN", F.col("DEP_DEL15")))
                            .over(window_route),
                          F.lit(0.15)
                      ))
    
    # 8. Southwest Relative Performance Index
    df = df.withColumn("sw_rel_perf",
                      F.when(F.col("origin_1yr_delay_rate") + F.lit(1e-6) > 0,
                            (F.col("sw_route_delay") + F.lit(1e-6)) 
                            / (F.col("origin_1yr_delay_rate") + F.lit(1e-6)))
                       .otherwise(1.0))
    
    # 9. Add route column
    df = df.withColumn("route", F.concat_ws("-", "ORIGIN", "DEST"))
    
    return df.drop("days_since_epoch")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Time-Based Profile Features

# COMMAND ----------

def add_time_based_features(df):
    """
    Enhanced time-based features with 15-minute buckets and robust null handling
    """
    print("Adding time-based features...")
    
    # 1. Time Bucket Features
    df = (df
          .withColumn("dep_hour", (F.col("CRS_DEP_TIME") / 100).cast("int"))
          .withColumn("dep_minute", (F.col("CRS_DEP_TIME") % 100))
          .withColumn("time_bucket_15min", 
                     (F.col("dep_minute") / 15).cast("int"))
          .withColumn("time_bucket", 
                     F.concat_ws("_", "dep_hour", "time_bucket_15min"))
         )
    
    # 2. Time Categories
    df = (df
          .withColumn("time_of_day_category",
                     F.when(F.col("dep_hour").between(5, 9), "morning")
                      .when(F.col("dep_hour").between(10, 15), "midday")
                      .when(F.col("dep_hour").between(16, 19), "evening")
                      .otherwise("night"))
          .withColumn("is_weekend", F.when(F.col("DAY_OF_WEEK").isin(6, 7), 1).otherwise(0))
          .withColumn("holiday_season", F.when(F.col("MONTH").isin(6, 7, 12), 1).otherwise(0))
         )
    
    # 3. Daily Operations (prerequisite for other features)
    window_daily = Window.partitionBy("ORIGIN", "FL_DATE")
    df = df.withColumn("daily_operations", F.count("*").over(window_daily))
    
    # 4. Prior Day Delay Rate with Hierarchical Fallbacks
    window_prior_day = Window.partitionBy("ORIGIN").orderBy("FL_DATE")
    window_origin_avg = Window.partitionBy("ORIGIN")
    
    df = (df
          .withColumn("daily_delay_rate",
                     F.avg(F.col("DEP_DEL15")).over(window_daily))
          .withColumn("prior_day_delay_rate",
                     F.coalesce(
                         F.lag("daily_delay_rate").over(window_prior_day),
                         F.avg(F.col("DEP_DEL15")).over(window_origin_avg),
                         F.lit(0.15)  # Global average fallback
                     ))
         )
    
    # 5. Same-Day Prior Delays with Zero-Count Protection
    window_same_day = Window.partitionBy("ORIGIN", "FL_DATE").orderBy("CRS_DEP_TIME")
    df = (df
          .withColumn("prior_flights_today",
                     F.coalesce(
                         F.count("*").over(window_same_day.rowsBetween(Window.unboundedPreceding, -1)),
                         F.lit(0)
                     ))
          .withColumn("prior_delays_today",
                     F.coalesce(
                         F.sum(F.col("DEP_DEL15")).over(window_same_day.rowsBetween(Window.unboundedPreceding, -1)),
                         F.lit(0)
                     ))
          .withColumn("same_day_prior_delay_percentage",
                     (F.col("prior_delays_today") + F.lit(0.5)) / 
                     (F.col("prior_flights_today") + F.lit(1)))
         )
    
    # 6. Time-Based Congestion Ratio with 3-Level Fallback
    window_historical = Window.partitionBy("ORIGIN", "DAY_OF_WEEK", "dep_hour") \
        .orderBy("FL_DATE") \
        .rowsBetween(-365, -1)  # Exclude current day
    
    df = (df
          .withColumn("avg_flights_per_dow_hour",
                     F.coalesce(
                         F.avg("daily_operations").over(window_historical),
                         F.avg("daily_operations").over(Window.partitionBy("ORIGIN")),
                         F.lit(10)  # Default small airport capacity
                     ))
          .withColumn("time_based_congestion_ratio",
                     F.when(F.col("avg_flights_per_dow_hour") > 0,
                           F.col("daily_operations") / F.col("avg_flights_per_dow_hour"))
                      .otherwise(1.0))
         )
    
    # Cleanup temporary columns
    return df.drop("dep_minute", "time_bucket_15min", "daily_delay_rate")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Weather-Based Profile Features

# COMMAND ----------

def add_weather_based_features(df):
    """
    Enhanced weather features with data-driven thresholds and statistical weighting
    """
    print("Adding weather-based features...")
    
    # 1. Extreme Weather Indicators (Dynamic Percentile Thresholds)
    weather_metrics = ["HourlyPrecipitation", "HourlyWindSpeed", 
                      "HourlyDryBulbTemperature", "HourlyVisibility"]
    
    percentiles = {metric: df.approxQuantile(metric, [0.05, 0.95], 0.05)
                   for metric in weather_metrics}
    
    df = (df
          .withColumn("extreme_precipitation", F.when(F.col("HourlyPrecipitation") >= percentiles["HourlyPrecipitation"][1], 1).otherwise(0))
          .withColumn("extreme_wind", F.when(F.col("HourlyWindSpeed") >= percentiles["HourlyWindSpeed"][1], 1).otherwise(0))
          .withColumn("extreme_temperature", F.when(
              (F.col("HourlyDryBulbTemperature") <= percentiles["HourlyDryBulbTemperature"][0]) |
              (F.col("HourlyDryBulbTemperature") >= percentiles["HourlyDryBulbTemperature"][1]), 1).otherwise(0))
          .withColumn("low_visibility", F.when(F.col("HourlyVisibility") <= percentiles["HourlyVisibility"][0], 1).otherwise(0))
         )
    
    # 2. Empirical Risk Score (Explicit type handling)
    delay_rates = {}
    for condition in ["extreme_precipitation", "extreme_wind", 
                     "extreme_temperature", "low_visibility"]:
        # Handle potential None values explicitly
        condition_agg = df.filter(F.col(condition) == 1).agg(F.avg("DEP_DEL15").alias("rate"))
        condition_rate = condition_agg.first()["rate"] or 0.15
        
        baseline_agg = df.filter(F.col(condition) == 0).agg(F.avg("DEP_DEL15").alias("rate")) 
        baseline_rate = baseline_agg.first()["rate"] or 0.15
        
        # Explicit type conversion and handling
        impact = float(condition_rate) - float(baseline_rate)
        delay_rates[condition] = impact if impact > 0 else 0.0
    
    # Normalize weights
    total_impact = sum(delay_rates.values())
    weights = {k: (v/total_impact if total_impact > 0 else 0.25) 
              for k, v in delay_rates.items()}
    
    df = df.withColumn(
        "extreme_weather_score",
        (F.col("extreme_precipitation") * weights["extreme_precipitation"] +
         F.col("extreme_wind") * weights["extreme_wind"] +
         F.col("extreme_temperature") * weights["extreme_temperature"] +
         F.col("low_visibility") * weights["low_visibility"])
    )
    
    # 3. Heat Index Calculation (No changes needed)
    df = df.withColumn(
        "heat_index",
        F.when(
            (F.col("HourlyDryBulbTemperature") >= 80) &
            (F.col("HourlyRelativeHumidity") >= 40),
            -42.379 + 2.04901523*F.col("HourlyDryBulbTemperature") +
            10.14333127*F.col("HourlyRelativeHumidity") -
            0.22475541*F.col("HourlyDryBulbTemperature")*F.col("HourlyRelativeHumidity") -
            6.83783e-3*F.pow(F.col("HourlyDryBulbTemperature"), 2) -
            5.481717e-2*F.pow(F.col("HourlyRelativeHumidity"), 2) +
            1.22874e-3*F.pow(F.col("HourlyDryBulbTemperature"), 2)*F.col("HourlyRelativeHumidity") +
            8.5282e-4*F.col("HourlyDryBulbTemperature")*F.pow(F.col("HourlyRelativeHumidity"), 2) -
            1.99e-6*F.pow(F.col("HourlyDryBulbTemperature"), 2)*F.pow(F.col("HourlyRelativeHumidity"), 2)
        ).otherwise(F.col("HourlyDryBulbTemperature"))
    )
    
    # 4. Weather Volatility Metrics (Proper PySpark functions)
    window_weather = Window.partitionBy("ORIGIN").orderBy("FL_DATE", "CRS_DEP_TIME").rowsBetween(-24*7, 0)
    
    for metric in ["HourlyDryBulbTemperature", "HourlyWindSpeed"]:
        df = (df
              .withColumn(f"{metric}_mean", F.avg(metric).over(window_weather))
              .withColumn(f"{metric}_std", F.stddev(metric).over(window_weather))
              .withColumn(f"{metric}_z", 
                         (F.col(metric) - F.col(f"{metric}_mean")) / F.col(f"{metric}_std"))
             )
    
    df = df.withColumn(
        "rapid_weather_change",
        F.when(
            (F.abs(F.col("HourlyDryBulbTemperature_z")) > 3) |
            (F.abs(F.col("HourlyWindSpeed_z")) > 3), 1
        ).otherwise(0)
    )
    
    # 5. Standardized Weather Anomalies
    window_seasonal = Window.partitionBy("ORIGIN", "MONTH")
    
    df = df.withColumn(
        "temp_anomaly_z",
        F.when(
            (F.col("HourlyDryBulbTemperature") - F.avg("HourlyDryBulbTemperature").over(window_seasonal)) / 
            F.stddev("HourlyDryBulbTemperature").over(window_seasonal) > 0,
            (F.col("HourlyDryBulbTemperature") - F.avg("HourlyDryBulbTemperature").over(window_seasonal)) / 
            F.stddev("HourlyDryBulbTemperature").over(window_seasonal)
        ).otherwise(0.0)
    )
    
    return df
    
    # Cleanup intermediate columns
    columns_to_drop = [f"{metric}_{stat}" 
                      for metric in ["HourlyDryBulbTemperature", "HourlyWindSpeed"]
                      for stat in ["mean", "std", "z"]]
    
    return df.drop(*columns_to_drop)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Southwest Airlines Profile Features

# COMMAND ----------

def add_southwest_profile_features(df):
    """
    Fixed Southwest features with proper windowing and global aggregations
    """
    print("Adding Southwest Airlines profile features...")
    
    # Calculate global percentiles once
    global_delay_median = df.approxQuantile("DEP_DEL15", [0.5], 0.05)[0]
    global_delay_p30 = df.approxQuantile("DEP_DEL15", [0.3], 0.05)[0]
    
    # Calculate global averages for route importance
    total_flights = df.count()
    avg_distance = df.agg(F.avg("DISTANCE")).first()[0]

    # 1. Time-of-Day Performance
    window_time_expanding = Window.partitionBy("ORIGIN", "time_of_day_category") \
        .orderBy("FL_DATE") \
        .rowsBetween(Window.unboundedPreceding, -1)
    
    df = df.withColumn(
        "sw_time_of_day_delay_rate",
        F.coalesce(
            F.avg(F.when(F.col("OP_CARRIER") == "WN", F.col("DEP_DEL15")))
              .over(window_time_expanding),
            F.avg(F.col("DEP_DEL15")).over(Window.partitionBy("ORIGIN")),
            F.lit(global_delay_median)
        )
    )
    
    # 2. Day-of-Week Performance
    window_dow = Window.partitionBy("ORIGIN", "DEST", "DAY_OF_WEEK")
    df = df.withColumn(
        "sw_day_of_week_delay_rate",
        (F.sum(F.col("DEP_DEL15")).over(window_dow) + 3*global_delay_p30) / 
        (F.count("*").over(window_dow) + 3)
    )
    
    # 3. Aircraft Performance
    window_aircraft = Window.partitionBy("TAIL_NUM")
    window_route = Window.partitionBy("ORIGIN", "DEST")
    df = df.withColumn(
        "sw_aircraft_delay_rate",
        F.coalesce(
            F.avg(F.col("DEP_DEL15")).over(window_aircraft),
            F.avg(F.col("DEP_DEL15")).over(window_route),
            F.lit(global_delay_median)
        )
    )
    
    # 4. Dynamic Hub Identification
    hub_threshold_df = df.filter(F.col("OP_CARRIER") == "WN") \
        .groupBy("ORIGIN") \
        .agg(F.count("*").alias("sw_volume"))
    
    hub_threshold = hub_threshold_df.approxQuantile("sw_volume", [0.85], 0.05)[0]
    hubs = hub_threshold_df.filter(F.col("sw_volume") >= hub_threshold) \
                         .select("ORIGIN").distinct().rdd.flatMap(lambda x: x).collect()
    
    df = df.withColumn("sw_origin_hub", F.col("ORIGIN").isin(hubs).cast("int"))
    
    # 5. Schedule Buffer Ratio
    window_route_history = Window.partitionBy("ORIGIN", "DEST") \
        .orderBy("FL_DATE") \
        .rowsBetween(-365, -1)
    
    df = df.withColumn(
        "sw_schedule_buffer_ratio",
        F.coalesce(
            F.col("CRS_ELAPSED_TIME") / F.avg("CRS_ELAPSED_TIME").over(window_route_history),
            F.lit(1.0)
        )
    )
    
    # 6. Origin-Time Performance
    window_origin_time = Window.partitionBy("ORIGIN", "time_bucket")
    df = df.withColumn(
        "sw_origin_time_perf",
        F.coalesce(
            F.avg(F.col("DEP_DEL15")).over(window_origin_time),
            F.col("sw_time_of_day_delay_rate"),
            F.lit(global_delay_median)
        )
    )
    
    # 7. Route Importance Score
    route_window = Window.partitionBy("ORIGIN", "DEST")
    df = df.withColumn(
        "sw_route_importance",
        (F.count("*").over(route_window) / F.lit(total_flights)) + 
        (F.avg("DISTANCE").over(route_window) / F.lit(avg_distance))
    )
    
    return df

# COMMAND ----------

def add_all_features(df):
    """
    Master function to add all engineered features to the DataFrame.
    
    Args:
        df: PySpark DataFrame with flight data
        
    Returns:
        DataFrame with all added features
    """
    print("Starting feature engineering process...")
    
    # Apply each feature set in sequence
    print("Adding airport profile features...")
    df = add_airport_profile_features(df)
    
    print("Adding time-based features...")
    df = add_time_based_features(df)
    
    # print("Adding weather-based features...")
    # df = add_weather_based_features(df)
    
    print("Adding Southwest Airlines profile features...")
    df = add_southwest_profile_features(df)
    
    print("Feature engineering complete!")
    
    # Calculate the number of features added
    original_cols = set(df.columns)
    engineered_cols = [col for col in df.columns if col not in original_cols]
    print(f"Added {len(engineered_cols)} new features to the DataFrame.")
    
    return df


# COMMAND ----------

df_otpw_60M_complete_parquet_cleaned.columns

# COMMAND ----------

# Call the master function on your cleaned DataFrame
df_otpw_60M_complete_parquet_with_features = add_all_features(df_otpw_60M_complete_parquet_cleaned)

# Display the schema of the final DataFrame to verify all features were added
print("Schema of final DataFrame with all features:")
df_otpw_60M_complete_parquet_with_features.printSchema()

# COMMAND ----------

# Count rows and columns in the final DataFrame
row_count = df_otpw_60M_complete_parquet_with_features.count()
col_count = len(df_otpw_60M_complete_parquet_with_features.columns)
print(f"Final DataFrame contains {row_count:,} rows and {col_count} columns.")

# COMMAND ----------

folder_path = f"dbfs:/student-groups/Group_04_04"

# Save the final DataFrame for future use
# df_otpw_60M_complete_parquet_with_features.write.parquet(f"{folder_path}/df_otpw_60M_complete_parquet_with_features_custom_join.parquet")
df_otpw_60M_complete_parquet_with_features.write.parquet(f"{folder_path}/df_otpw_60M_complete_parquet_with_features_custom_join_2015_2021.parquet")
# df_otpw_60M_complete_parquet_with_features.write.mode('overwrite').parquet(f"{folder_path}/df_otpw_60M_complete_parquet_with_features_custom_join.parquet")

# COMMAND ----------

data_BASE_DIR = "dbfs:/student-groups/Group_04_04/"
display(dbutils.fs.ls(f"{data_BASE_DIR}"))

# COMMAND ----------

# df_otpw_60M_complete_parquet_with_additional_features = spark.read.parquet(f"dbfs:/student-groups/Group_04_04/df_otpw_60M_complete_parquet_with_features_custom_join_2015_2021.parquet/")
df_otpw_60M_complete_parquet_with_additional_features = df_otpw_60M_complete_parquet_with_features

# COMMAND ----------

display(df_otpw_60M_complete_parquet_with_additional_features)

# COMMAND ----------

# Fill nulls in specific anomaly columns
df_otpw_60M_complete_parquet_with_additional_features = (
    df_otpw_60M_complete_parquet_with_additional_features
    .withColumn("temp_anomaly_z", F.coalesce(F.col("temp_anomaly_z"), F.lit(0.0)))
    # .withColumn("precip_anomaly_z", F.coalesce(F.col("precip_anomaly_z"), F.lit(0.0)))
)

# COMMAND ----------

# Update DEP_DEL15 column: change 2 (cancelled) to 1 (delayed)
df_otpw_60M_complete_parquet_with_additional_features_modified = df_otpw_60M_complete_parquet_with_additional_features.withColumn(
    "DEP_DEL15",
    F.when(F.col("DEP_DEL15") == 2, 1).otherwise(F.col("DEP_DEL15"))
)

# Verify the change
cancelled_count = df_otpw_60M_complete_parquet_with_additional_features_modified.filter(F.col("DEP_DEL15") == 2).count()
print(f"Number of flights still marked as cancelled (should be 0): {cancelled_count}")

# COMMAND ----------

len(df_otpw_60M_complete_parquet_with_additional_features_modified.columns)

# COMMAND ----------

# Filter for Southwest Airlines flights only (carrier code "WN")
df_southwest = df_otpw_60M_complete_parquet_with_additional_features_modified.filter(
    df_otpw_60M_complete_parquet_with_additional_features["OP_CARRIER"] == "WN"
)

# Check the number of rows after filtering
southwest_count = df_southwest.count()
print(f"Number of Southwest Airlines flights: {southwest_count:,}")


# COMMAND ----------

# Analyze missing values in cleaned features
null_percentage_df = calculate_null_percentage(df_southwest)
display(null_percentage_df)

# COMMAND ----------

# Get the current DataFrame schema and column count before dropping
print("Original DataFrame column count:", len(df_southwest.columns))

# List of columns to drop
columns_to_drop = [
    "FL_DATE", "OP_UNIQUE_CARRIER", "OP_CARRIER_AIRLINE_ID", "OP_CARRIER", "TAIL_NUM", "OP_CARRIER_FL_NUM",
    "ORIGIN_AIRPORT_SEQ_ID", "ORIGIN_CITY_MARKET_ID", "ORIGIN", "ORIGIN_CITY_NAME",
    "ORIGIN_STATE_FIPS", "ORIGIN_STATE_NM", "ORIGIN_WAC",
    "DEST_AIRPORT_SEQ_ID", "DEST_CITY_MARKET_ID", "DEST", "DEST_CITY_NAME",
    "DEST_STATE_FIPS", "DEST_STATE_NM", "DEST_WAC", 
    "DEP_TIME_BLK", "CRS_ARR_TIME", "ARR_TIME_BLK", "CANCELLED", "DIVERTED",
    "FLIGHTS", "DIV_AIRPORT_LANDINGS",
    "sched_depart_date_time", "sched_depart_date_time_UTC", "depart_date_time", "depart_date_time_UTC",
    "arrival_date_time_UTC", "four_hours_prior_depart_UTC", "two_hours_prior_depart_UTC",
    "origin_airport_name", "origin_station_id", "origin_station_name", "origin_dis",
    "dest_airport_name", "dest_station_id", "dest_station_name", "dest_dis",
    "PREV_ORIGIN", "PREV_arrival_date_time_UTC", "PREV_ARR_DELAY", "PREV_ARR_DELAY_NEW",
    "STATION", "LATITUDE", "LONGITUDE", "ELEVATION", "NAME", "REPORT_TYPE", "SOURCE", "REM",
    "WindEquipmentChangeDate", "DATE"
]

# Drop the specified columns
df_southwest_reduced = df_southwest.drop(*columns_to_drop)

# Display the remaining column count
print("Reduced DataFrame column count:", len(df_southwest_reduced.columns))

# Display the remaining columns for reference
print("\nRemaining columns:")
for col in df_southwest_reduced.columns:
    print(col)


# COMMAND ----------

len(df_southwest_reduced.columns)

# COMMAND ----------

from pyspark.sql.functions import col, when, isnan

# Convert the 'time_bucket' column to a numeric type
df_southwest_reduced = df_southwest_reduced.withColumn("time_bucket", 
                                                     df_southwest_reduced["time_bucket"].cast("double"))

# List of numeric columns
numeric_cols = [
    "QUARTER", "MONTH", "DAY_OF_MONTH", "DAY_OF_WEEK", "ORIGIN_AIRPORT_ID", "DEST_AIRPORT_ID",
    "DEP_DEL15", "CRS_ELAPSED_TIME", "DISTANCE", "DISTANCE_GROUP", "YEAR",
    "origin_airport_lat", "origin_airport_lon", "dest_airport_lat", "dest_airport_lon",
    "PREV_CANCELLED", "MINUTES_BETWEEN_FLIGHTS", "PREV_ARR_DEL15",
    "HourlyAltimeterSetting", "HourlyDewPointTemperature", "HourlyDryBulbTemperature",
    "HourlyPrecipitation", "HourlyRelativeHumidity", "HourlySeaLevelPressure",
    "HourlyStationPressure", "HourlyVisibility", "HourlyWetBulbTemperature",
    "HourlyWindDirection", "HourlyWindSpeed", "daily_operations", "rolling_30day_volume",
    "origin_1yr_delay_rate", "route_1yr_volume", "sw_market_share", "sw_30d_delay",
    "sw_route_delay", "sw_rel_perf", "dep_hour", "is_weekend",
    "holiday_season", "prior_day_delay_rate", "prior_flights_today", "prior_delays_today",
    "same_day_prior_delay_percentage", "avg_flights_per_dow_hour",
    "time_based_congestion_ratio", "extreme_precipitation", "extreme_wind",
    "extreme_temperature", "low_visibility", "extreme_weather_score", "heat_index",
    "rapid_weather_change", "temp_anomaly_z",
    "sw_time_of_day_delay_rate", "sw_day_of_week_delay_rate", "sw_aircraft_delay_rate",
    "sw_origin_hub", "sw_schedule_buffer_ratio", "sw_origin_time_perf", "sw_route_importance"
]

# First, handle null values in all numeric columns
for col_name in numeric_cols:
    df_southwest_reduced = df_southwest_reduced.withColumn(
        col_name,
        when(col(col_name).isNull() | isnan(col(col_name)), 0).otherwise(col(col_name))
    )

# Now create the vector assembler with handleInvalid="skip"
vector_col = "correlation_features"
assembler = VectorAssembler(inputCols=numeric_cols, outputCol=vector_col, handleInvalid="skip")
df_vector = assembler.transform(df_southwest_reduced).select(vector_col)

# Compute correlation matrix
matrix = Correlation.corr(df_vector, vector_col).collect()[0][0]
corr_matrix = matrix.toArray().tolist()

# Convert to pandas DataFrame for visualization
corr_df = pd.DataFrame(corr_matrix, columns=numeric_cols, index=numeric_cols)

# Create a heatmap
plt.figure(figsize=(20, 16))
sns.heatmap(corr_df, annot=False, cmap="coolwarm", vmin=-1, vmax=1, center=0)
plt.title("Correlation Matrix Heatmap")
plt.tight_layout()
display(plt.gcf())  # Display the figure in Databricks

# Save the figure if needed
plt.savefig("/dbfs/FileStore/pearson_correlation_heatmap.png", dpi=300, bbox_inches="tight")
plt.close()

# Identify highly correlated features
threshold = 0.8
high_corr = []
for i in range(len(numeric_cols)):
    for j in range(i+1, len(numeric_cols)):
        if abs(corr_matrix[i][j]) > threshold:
            high_corr.append((numeric_cols[i], numeric_cols[j], corr_matrix[i][j]))

print("\nHighly correlated features (correlation > 0.8):")
for feat1, feat2, corr in sorted(high_corr, key=lambda x: abs(x[2]), reverse=True):
    print(f"{feat1} - {feat2}: {corr:.3f}")

# COMMAND ----------

# Create year-quarter and quarter-month combined columns
df_reduced = df_southwest_reduced.withColumn(
    "year_quarter", 
    concat(col("YEAR").cast("string"), lit("-Q"), col("QUARTER").cast("string"))
).withColumn(
    "quarter_month", 
    concat(col("QUARTER").cast("string"), lit("-"), col("MONTH").cast("string"))
)

# COMMAND ----------

# List of columns to drop
columns_to_drop = [
    "DISTANCE", 
    "low_visibility", 
    "HourlyDryBulbTemperature", 
    "DISTANCE_GROUP", 
    "CRS_ELAPSED_TIME", 
    "QUARTER", 
    "HourlyAltimeterSetting", 
    "avg_flights_per_dow_hour", 
    "HourlyWetBulbTemperature", 
    "HourlyDewPointTemperature",
    "HourlyDryBulbTemperature" 
]

# Drop the columns
df_final = df_reduced.drop(*columns_to_drop)

# Check the number of remaining columns
remaining_columns = len(df_final.columns)
print(f"Number of columns after dropping redundant features: {remaining_columns}")

# COMMAND ----------

df_final.columns

# COMMAND ----------

folder_path = f"dbfs:/student-groups/Group_04_04"

# Save df_southwest_cleaned as a parquet file
df_final.write.parquet(f"{folder_path}/df_otpw_60M_complete_parquet_final_custom_join_2015-2021.parquet") # use to save a new version
# df_final.write.mode('overwrite').parquet(f"{folder_path}/df_otpw_60M_complete_parquet_final_custom_join.parquet")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Start Data Modelling

# COMMAND ----------

# df_final = spark.read.parquet(f"dbfs:/student-groups/Group_04_04/df_otpw_60M_complete_parquet_final_custom_join.parquet/")
df_final = spark.read.parquet(f"dbfs:/student-groups/Group_04_04/df_otpw_60M_complete_parquet_final_custom_join_with_graph_features.parquet/")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Graph features

# COMMAND ----------

# years for PageRank
years = [row.YEAR for row in df_final.select("YEAR").distinct().collect()]
years.sort()
print(years)

# all year_quarter values for InDegree/OutDegree
year_quarters = [
    row["year_quarter"] 
    for row in df_final.select("year_quarter").distinct().collect()
]
year_quarters.sort()



# COMMAND ----------

# MAGIC %md
# MAGIC ##### PageRank

# COMMAND ----------


all_pagerank_years = []  # list for pagerank for each year

for y in years:
    print(f"Computing PageRank for year = {y}")
    #Filter DataFrame to this year
    df_year = df_final.filter(F.col("YEAR") == y)

    # Create Vertices (Airports)
    # origin side
    origin_airports = (
        df_year.selectExpr(
            "ORIGIN_AIRPORT_ID as airport_id",
            "ORIGIN_STATE_ABR as state_abbr",
            "origin_airport_lat as lat",
            "origin_airport_lon as lon",
            "origin_type as airport_type"
        )
        .dropDuplicates(["airport_id"])
        .withColumn("priority", F.lit(1))
    )

    # destination side
    dest_airports = (
        df_year.selectExpr(
            "DEST_AIRPORT_ID as airport_id",
            "DEST_STATE_ABR as state_abbr",
            "dest_airport_lat as lat",
            "dest_airport_lon as lon",
            "dest_type as airport_type"
        )
        .dropDuplicates(["airport_id"])
        .withColumn("priority", F.lit(2))
    )

    union_airports = origin_airports.union(dest_airports)

    window = Window.partitionBy("airport_id").orderBy("priority")
    all_airports = (
        union_airports
        .withColumn("row_num", F.row_number().over(window))
        .filter("row_num = 1")
        .drop("row_num", "priority")
    )

    # renaming 'id' for GraphFrames
    airports_vertices = all_airports.selectExpr(
        "airport_id as id",
        "state_abbr",
        "lat",
        "lon",
        "airport_type"
    )
    # creating edges dataframe
    # Weighted by avg prior_day_delay_rate for each (origin, dest) in this year
    flight_edges = (
        df_year
        .groupBy("ORIGIN_AIRPORT_ID", "DEST_AIRPORT_ID")
        .agg(F.avg("prior_day_delay_rate").alias("weight"))
        .selectExpr(
            "ORIGIN_AIRPORT_ID as src",
            "DEST_AIRPORT_ID as dst",
            "weight"
        )
    )

    # Run PageRank for each year
    g = GraphFrame(airports_vertices, flight_edges)
    results = g.pageRank(
        resetProbability=0.15,
        tol=0.0001,

    )
    print(f"Airports: {g.vertices.count()}")
    print(f"Trips: {g.edges.count()}")   

    # results (id, pagerank)
    pagerank_vertices = results.vertices.select("id", "pagerank")

    
    #Store This Year's PageRank
    pagerank_for_year = pagerank_vertices.withColumn("YEAR", F.lit(y))
    all_pagerank_years.append(pagerank_for_year)


# COMMAND ----------

if all_pagerank_years:
    pagerank_all_years = all_pagerank_years[0]
    for df_rest in all_pagerank_years[1:]:
        pagerank_all_years = pagerank_all_years.union(df_rest)
else:
    pagerank_all_years = spark.createDataFrame([], "id string, pagerank double, YEAR int")

# lag window
window_spec = Window.partitionBy("id").orderBy("YEAR")

pagerank_all_years_lag = pagerank_all_years.withColumn(
    "pagerank_lag",
    F.lag("pagerank").over(window_spec)
)

pagerank_all_years_lag.show(10)

# COMMAND ----------

# rename columns
pagerank_origin = (
    pagerank_all_years_lag
    .withColumnRenamed("id", "ORIGIN_AIRPORT_ID")
    .withColumnRenamed("pagerank", "pagerank_origin")
    .withColumnRenamed("pagerank_lag", "pagerank_origin_lag")
)

# Similarly for destination side
pagerank_destination = (
    pagerank_all_years_lag
    .withColumnRenamed("id", "DEST_AIRPORT_ID")
    .withColumnRenamed("pagerank", "pagerank_destination")
    .withColumnRenamed("pagerank_lag", "pagerank_destination_lag")
)

# Now join them onto your main flights DataFrame
graph_df = (
    df_final
    .join(
        pagerank_origin,
        on=["ORIGIN_AIRPORT_ID", "YEAR"],  # match both airport and year
        how="left"
    )
    .join(
        pagerank_destination,
        on=["DEST_AIRPORT_ID", "YEAR"],
        how="left"
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### InDegree/OutDegree

# COMMAND ----------

# List to store computed per-quarter degree DataFrames
all_degree_quarters = []

# Loop over all quarters to compute the degree values from that quarter’s flight data.
for q in year_quarters:
    print(f"Computing in/out degrees for quarter {q}")
    
    # Filter data for quarter q
    df_q = df_final.filter(F.col("year_quarter") == q)
    
    # Build vertices: get distinct airports from both origins and destinations
    origin_airports = (
        df_q.selectExpr("ORIGIN_AIRPORT_ID as airport_id")
        .dropDuplicates(["airport_id"])
        .withColumn("priority", F.lit(1))
    )
    dest_airports = (
        df_q.selectExpr("DEST_AIRPORT_ID as airport_id")
        .dropDuplicates(["airport_id"])
        .withColumn("priority", F.lit(2))
    )
    union_airports = origin_airports.union(dest_airports)
    
    # Remove duplicates: keep one row per airport
    quarter_window = Window.partitionBy("airport_id").orderBy("priority")
    all_airports = (
        union_airports
        .withColumn("row_num", F.row_number().over(quarter_window))
        .filter("row_num = 1")
        .drop("row_num", "priority")
    )
    
    # Rename column to "id" to use as vertices in GraphFrame
    airports_vertices = all_airports.selectExpr("airport_id as id")
    
    # Build edges for the graph (using flights within quarter q)
    flight_edges = df_q.selectExpr(
        "ORIGIN_AIRPORT_ID as src",
        "DEST_AIRPORT_ID as dst"
    )
    
    # Create the GraphFrame
    g = GraphFrame(airports_vertices, flight_edges)
    
    # Compute inDegrees and outDegrees
    inDegDF = g.inDegrees 
    outDegDF = g.outDegrees
    
    # Join in/out degree DataFrames, filling missing values with 0
    degreeDF = (
        inDegDF.join(outDegDF, on="id", how="full")
        .na.fill(0, ["inDegree", "outDegree"])
    )
    
    # Attach quarter label q to the computed degree values
    degree_quarter = degreeDF.withColumn("year_quarter", F.lit(q))
    all_degree_quarters.append(degree_quarter)

# Union all quarter DataFrames.
deg_all = all_degree_quarters[0]
for df in all_degree_quarters[1:]:
    deg_all = deg_all.union(df)

windowSpec = Window.partitionBy("id").orderBy("year_quarter")

# Create lag columns for inDegree and outDegree; for the earliest quarter (e.g. Q1)
lagged = deg_all.withColumn("lag_inDegree", F.lag("inDegree", 1).over(windowSpec)) \
                .withColumn("lag_outDegree", F.lag("outDegree", 1).over(windowSpec))


# COMMAND ----------

origin_degrees = (
    lagged
    .withColumnRenamed("id", "ORIGIN_AIRPORT_ID")
    .withColumnRenamed("inDegree", "origin_indegree")
    .withColumnRenamed("outDegree", "origin_outdegree")
    .withColumnRenamed("lag_inDegree", "origin_indegree_lag")
    .withColumnRenamed("lag_outDegree", "origin_outdegree_lag")
)

dest_degrees = (
    lagged
    .withColumnRenamed("id", "DEST_AIRPORT_ID")
    .withColumnRenamed("inDegree", "dest_indegree")
    .withColumnRenamed("outDegree", "dest_outdegree")
    .withColumnRenamed("lag_inDegree", "dest_indegree_lag")
    .withColumnRenamed("lag_outDegree", "dest_outdegree_lag")
)

graph_df = (
    graph_df
    .join(origin_degrees, on=["ORIGIN_AIRPORT_ID", "year_quarter"], how="left")
    .join(dest_degrees, on=["DEST_AIRPORT_ID", "year_quarter"], how="left")
)


# COMMAND ----------

# MAGIC %md
# MAGIC ##### Imputations

# COMMAND ----------

# Fill null values in pagerank lag columns
graph_df = graph_df.withColumn(
    "pagerank_origin_lag",
    when(col("pagerank_origin_lag").isNull(), col("pagerank_origin"))
    .otherwise(col("pagerank_origin_lag"))
)

graph_df = graph_df.withColumn(
    "pagerank_destination_lag",
    when(col("pagerank_destination_lag").isNull(), col("pagerank_destination"))
    .otherwise(col("pagerank_destination_lag"))
)

# Fill null values in degree lag columns (in_degree and out_degree)
graph_df = graph_df.withColumn(
    "origin_indegree_lag",
    when(col("origin_indegree_lag").isNull(), col("origin_indegree"))
    .otherwise(col("origin_indegree_lag"))
)

graph_df = graph_df.withColumn(
    "origin_outdegree_lag",
    when(col("origin_outdegree_lag").isNull(), col("origin_outdegree"))
    .otherwise(col("origin_outdegree_lag"))
)

graph_df = graph_df.withColumn(
    "dest_indegree_lag",
    when(col("dest_indegree_lag").isNull(), col("dest_indegree"))
    .otherwise(col("dest_indegree_lag"))
)

graph_df = graph_df.withColumn(
    "dest_outdegree_lag",
    when(col("dest_outdegree_lag").isNull(), col("dest_outdegree"))
    .otherwise(col("dest_outdegree_lag"))
)

# COMMAND ----------

# columns_to_check = [
#     "pagerank_origin_lag",
#     "pagerank_destination_lag",
#     "origin_indegree_lag",
#     "origin_outdegree_lag",
#     "dest_indegree_lag",
#     "dest_outdegree_lag"
# ]

# null_counts = graph_df.select([F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in columns_to_check])
# display(null_counts)

# COMMAND ----------

df_final = graph_df

# COMMAND ----------

folder_path = f"dbfs:/student-groups/Group_04_04"

# Save df_southwest_cleaned as a parquet file
df_final.write.parquet(f"{folder_path}/df_otpw_60M_complete_parquet_final_custom_join_2015_2024.parquet") # use to save a new version
# df_final.write.mode('overwrite').parquet(f"{folder_path}/df_otpw_60M_complete_parquet_final_custom_join.parquet")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Exploratory Data Analysis on the New Features

# COMMAND ----------

df_final = spark.read.parquet(f"dbfs:/student-groups/Group_04_04/df_otpw_60M_complete_parquet_final_custom_join_2015_2024.parquet/")

# COMMAND ----------

# Convert to Pandas for plotting
df_pd = df_final.select(
    "daily_operations", "sw_rel_perf", "origin_type", "prior_day_delay_rate", 
    "same_day_prior_delay_percentage", "time_of_day_category", "DEP_DEL15", 
    "time_based_congestion_ratio", "pagerank_origin", "origin_outdegree",
    "sw_origin_time_perf", "sw_market_share"
).sample(fraction=0.1, seed=42).toPandas()

# COMMAND ----------

# Set up Matplotlib aesthetics
sns.set(style='whitegrid', palette='pastel')

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# SW Relative Performance (log-scale, remove outliers)
filtered_df = df_pd[df_pd['sw_rel_perf'] < df_pd['sw_rel_perf'].quantile(0.99)]
sns.scatterplot(data=filtered_df, x='daily_operations', y='sw_rel_perf', hue='origin_type', ax=axes[0])
axes[0].set_yscale('log')
axes[0].set_title('SW Relative Performance (Log scale) vs. Daily Operations (2015-2024)')

# SW Market Share (violin plot)
sns.violinplot(x=df_pd['origin_type'], y=df_pd['sw_market_share'], ax=axes[1])
axes[1].set_title('SW Market Share Distribution by Airport Type (2015-2024)')

plt.tight_layout()
plt.savefig('airport_profile.png')
plt.show()

# COMMAND ----------

# 2. Time-based Features EDA
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.regplot(
    data=df_pd, 
    x='prior_day_delay_rate', 
    y='same_day_prior_delay_percentage',
    scatter_kws={'alpha':0.3},
    line_kws={'color':'red'},
    ax=axes[0]
)
axes[0].set_title('Prior Day Delay Rate vs. Same Day Prior Delay Percentage (2015-2024)')
axes[0].set_xlabel('Prior Day Delay Rate')
axes[0].set_ylabel('Same Day Prior Delay %')

# Delay Rate by Time of Day Category
sns.barplot(data=df_pd, x='time_of_day_category', y='DEP_DEL15', ax=axes[1])
axes[1].set_title('Delay Rate by Time of Day Category (2015-2024)')
axes[1].set_xlabel('Time of Day')
axes[1].set_ylabel('Average Delay Rate')

plt.tight_layout()
plt.savefig('time_based_features.png')

# COMMAND ----------

fig, ax = plt.subplots(figsize=(10, 6))

sns.violinplot(x='origin_type', y='sw_origin_time_perf', data=df_pd)
plt.title("Southwest Airlines Performance by Airport Type (2015-2024)")

plt.tight_layout()
plt.savefig('sw_specific_features.png')

# COMMAND ----------

# Bin PageRank and OutDegree
df_pd['pagerank_bin'] = pd.qcut(df_pd['pagerank_origin'], q=10, duplicates='drop')
df_pd['outdegree_bin'] = pd.qcut(df_pd['origin_outdegree'], q=10, duplicates='drop')

# Group by bins
pagerank_grp = df_pd.groupby('pagerank_bin')['DEP_DEL15'].mean().reset_index()
outdegree_grp = df_pd.groupby('outdegree_bin')['DEP_DEL15'].mean().reset_index()

# Plot
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.barplot(x='pagerank_bin', y='DEP_DEL15', data=pagerank_grp, ax=axes[0])
axes[0].set_title('Avg Delay Rate by PageRank Bin (2015-2024)')
axes[0].set_xlabel('PageRank (Binned)')
axes[0].set_ylabel('Avg Delay Rate')
axes[0].tick_params(axis='x', rotation=45)

sns.barplot(x='outdegree_bin', y='DEP_DEL15', data=outdegree_grp, ax=axes[1])
axes[1].set_title('Avg Delay Rate by OutDegree Bin (2015-2024)')
axes[1].set_xlabel('OutDegree (Binned)')
axes[1].set_ylabel('Avg Delay Rate')
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('graph_based_features.png')

plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Spearman correlation for categorical features

# COMMAND ----------

# 1. Define categorical features
cat_features = [
    'ORIGIN_AIRPORT_ID', 'ORIGIN_STATE_ABR', 'DEST_AIRPORT_ID', 'DEST_STATE_ABR',
    'origin_type', 'dest_type', 'TRIPLET', 'route', 'time_bucket',
    'time_of_day_category', 'year_quarter', 'quarter_month'
]

# 2. Create pipeline stages for encoding
indexers = [StringIndexer(inputCol=col, outputCol=f"{col}_idx", handleInvalid="keep") 
           for col in cat_features]
    
# 3. Assemble vectors
assembler = VectorAssembler(
    inputCols=[f"{col}_idx" for col in cat_features],
    outputCol="features",
    handleInvalid="keep"
)

# 4. Build and run pipeline
pipeline = Pipeline(stages=indexers + [assembler])
indexed_data = pipeline.fit(df_final).transform(df_final)

# 5. Compute Spearman correlation matrix
correlation_matrix = Correlation.corr(indexed_data, "features", method="spearman").head()[0]

# 6. Convert to pandas/numpy for visualization
matrix_array = correlation_matrix.toArray()
correlation_df = pd.DataFrame(matrix_array, index=cat_features, columns=cat_features)

# 7. Plot heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_df, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Spearman Correlation Matrix for Categorical Features")
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Final features for modelling - picked based on EDA + Pearson Correlation + Spearman Correlation

# COMMAND ----------

selected_features_for_training = ['DAY_OF_MONTH',
 'DAY_OF_WEEK',
 'DEP_DEL15',
 'origin_type',
 'dest_type',
 'PREV_CANCELLED',
 'MINUTES_BETWEEN_FLIGHTS',
 'PREV_ARR_DEL15',
 'TRIPLET',
 'daily_operations',
 'rolling_30day_volume',
 'origin_1yr_delay_rate',
 'route_1yr_volume',
 'sw_market_share',
 'sw_30d_delay',
 'sw_route_delay',
 'sw_rel_perf',
 'route',
 'time_bucket',
 'time_of_day_category',
 'is_weekend',
 'holiday_season',
 'prior_day_delay_rate',
 'prior_flights_today',
 'prior_delays_today',
 'same_day_prior_delay_percentage',
 'time_based_congestion_ratio',
 'sw_time_of_day_delay_rate',
 'sw_day_of_week_delay_rate',
 'sw_aircraft_delay_rate',
 'sw_origin_hub',
 'sw_schedule_buffer_ratio',
 'sw_origin_time_perf',
 'sw_route_importance',
 'year_quarter',
 'quarter_month',
 'YEAR',
 'MONTH',
 'pagerank_origin_lag',
 'pagerank_destination_lag',
 'origin_indegree_lag',
 'origin_outdegree_lag',
 'dest_indegree_lag',
 'dest_outdegree_lag'
]

# COMMAND ----------

# Display the schema to verify the columns
print(f"Number of columns after filtering: {len(selected_features_for_training)}")

# COMMAND ----------

# Split the dataset FIRST to avoid leakage
test_data = df_final.filter(F.col("YEAR") == 2024)
validation_data = df_final.filter((F.col("YEAR") >= 2022) & (F.col("YEAR") <= 2023))
train_data = df_final.filter((F.col("YEAR") >= 2015) & (F.col("YEAR") <= 2021))

# Get number of cores
num_cores = spark.sparkContext.defaultParallelism
print(f"Number of current cores: {num_cores}")

# Repartition each set individually
train_data = train_data.repartition(num_cores * 2)#.persist()
validation_data = validation_data.repartition(num_cores)#.persist()
test_data = test_data.repartition(num_cores)#.persist()

# Force evaluation to materialize partitions and cache
# print(f"Train data count: {train_data.count()}")
# print(f"Validation data count: {validation_data.count()}")
# print(f"Test data count: {test_data.count()}")

# Split data using year_quarter column
# Train: 2015, Month = 1
# train_data = df_final.filter((col("YEAR") == "2015") & (col("MONTH") == 1))

# # Validation: 2015, Month = 2
# validation_data = df_final.filter((col("YEAR") == "2015") & (col("MONTH") == 2))

# # Test: 2015, Month = 3
# test_data = df_final.filter((col("YEAR") == "2015") & (col("MONTH") == 3))

# # Cache the datasets to improve performance
# train_data = train_data.cache()
# validation_data = validation_data.cache()
# test_data = test_data.cache()

# # Count records in each dataset
# print(f"Train data count: {train_data.count()}")
# print(f"Validation data count: {validation_data.count()}")
# print(f"Test data count: {test_data.count()}")

# COMMAND ----------

print(f"Train data count: {train_data.count()}")
print(f"Validation data count: {validation_data.count()}")
print(f"Test data count: {test_data.count()}")

# COMMAND ----------

# Store counts in variables
train_count = 9865375 
validation_count = 2735873
test_count = 1418940

# Use these variables instead of calling count() again
print(f"Train data count: {train_count}")
print(f"Validation data count: {validation_count}")
print(f"Test data count: {test_count}")

# COMMAND ----------

folder_path = f"dbfs:/student-groups/Group_04_04"

# Save train/validate/test split as a parquet files
# train_data.write.parquet(f"{folder_path}/df_otpw_60M_train_data.parquet") # use to save a new version
# validation_data.write.parquet(f"{folder_path}/df_otpw_60M_validation_data.parquet") # use to save a new version
# test_data.write.parquet(f"{folder_path}/df_otpw_60M_test_data.parquet") # use to save a new version


train_data.write.mode('overwrite').parquet(f"{folder_path}/df_otpw_2015_2024_train_data.parquet") # use if you want to overwrite exisiting file
validation_data.write.mode('overwrite').parquet(f"{folder_path}/df_otpw_2015_2024_validation_data.parquet") # use if you want to overwrite exisiting file
test_data.write.mode('overwrite').parquet(f"{folder_path}/df_otpw_2015_2024_test_data.parquet") # use if you want to overwrite exisiting file

# COMMAND ----------

train_data = spark.read.parquet(f"dbfs:/student-groups/Group_04_04/df_otpw_2015_2024_train_data.parquet/")
validation_data = spark.read.parquet(f"dbfs:/student-groups/Group_04_04/df_otpw_2015_2024_validation_data.parquet/")
test_data = spark.read.parquet(f"dbfs:/student-groups/Group_04_04/df_otpw_2015_2024_test_data.parquet/")

# COMMAND ----------

# Cache and trigger caching using .first()
train_data.cache().first()
validation_data.cache().first()
test_data.cache().first()


# COMMAND ----------

# Visualize delayed vs on-time flights for each dataset
plt.figure(figsize=(15, 5))

# Function to plot delayed vs on-time flights
def plot_delay_distribution(data, position, title, total_count=None):
    delay_counts = data.groupBy("DEP_DEL15").count().toPandas()
    delay_counts = delay_counts.sort_values("DEP_DEL15")
    
    # Use provided total count if available
    if total_count is None:
        total = delay_counts["count"].sum()
    else:
        total = total_count
    
    delay_counts["percentage"] = delay_counts["count"] / total * 100
    
    # Map 0/1 to meaningful labels
    delay_counts["status"] = delay_counts["DEP_DEL15"].map({0: "On-time", 1: "Delayed"})
    
    # Create subplot
    plt.subplot(1, 3, position)
    bars = plt.bar(delay_counts["status"], delay_counts["count"], color=["green", "red"])
    
    # Add count and percentage labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{height:,.0f}', ha='center', va='bottom')
    
    # Add percentage as a second line of text
    for i, p in enumerate(delay_counts["percentage"]):
        plt.text(bars[i].get_x() + bars[i].get_width()/2., bars[i].get_height()/2,
                f'{p:.1f}%', ha='center', va='center', color='white', fontweight='bold')
    
    plt.title(title)
    plt.ylabel("Number of Flights")
    plt.ylim(0, delay_counts["count"].max() * 1.1)

# Plot for each dataset
plot_delay_distribution(train_data, 1, "Train Data", train_count)
plot_delay_distribution(validation_data, 2, "Validation Data", validation_count)
plot_delay_distribution(test_data, 3, "Test Data", test_count)

plt.tight_layout()
plt.show()

# COMMAND ----------

# Use known delay rate (25%) instead of counting
known_delay_rate = 0.25  # 25% of flights are delayed
total_count = train_count  # Use your stored variable

# Calculate counts based on rate
delayed_count = int(total_count * known_delay_rate)
on_time_count = total_count - delayed_count

# Calculate weights
delayed_weight = total_count / (2 * delayed_count)
on_time_weight = total_count / (2 * on_time_count)

# Add weight column to training data
train_data = train_data.withColumn(
    "sample_weight",
    when(col("DEP_DEL15") == 1, delayed_weight).otherwise(on_time_weight)
)

# COMMAND ----------

# MAGIC %md
# MAGIC # MLLib pipeline starts

# COMMAND ----------

# MAGIC %md
# MAGIC Please note that for conveniece, we are evaluating test datasets for each model such that if one model is chosen, we don't need to run the costly model training and evaluation to see its test model, better efficiency and coordination. However, similar as what we did in cross validation, when evaluating the model, we only consider validation evaluation metrics.

# COMMAND ----------

# Define categorical features with few unique values (for one-hot encoding)
categorical_features = [
    # Airport identifiers
    'origin_type', 'dest_type',
    
    # Time-related categorical features
    'time_of_day_category', 'year_quarter', 'quarter_month',
    
    # Binary indicators
    'is_weekend', 'holiday_season', 'sw_origin_hub'
]

# Define high-cardinality categorical features (for numeric encoding only)
high_cardinality_features = [
    'route',         
    'time_bucket',  
    'TRIPLET'         
]

numerical_features = [
    # Time-related numeric features
    'DAY_OF_MONTH', 'DAY_OF_WEEK', 
    
    # Flight history features
    'PREV_CANCELLED', 'MINUTES_BETWEEN_FLIGHTS', 'PREV_ARR_DEL15',
    
    # Airport and route metrics
    'daily_operations', 'rolling_30day_volume', 'origin_1yr_delay_rate', 'route_1yr_volume',
    
    # Southwest Airlines specific metrics
    'sw_market_share', 'sw_30d_delay', 'sw_route_delay', 'sw_rel_perf',
    'sw_time_of_day_delay_rate', 'sw_day_of_week_delay_rate', 'sw_aircraft_delay_rate',
    'sw_schedule_buffer_ratio', 'sw_origin_time_perf', 'sw_route_importance',
    
    # Delay metrics
    'prior_day_delay_rate', 'prior_flights_today', 'prior_delays_today', 
    'same_day_prior_delay_percentage', 'time_based_congestion_ratio',

    # Graph based - PageRank
        'pagerank_origin_lag',
        'pagerank_destination_lag',
        'origin_indegree_lag',
        'origin_outdegree_lag',
        'dest_indegree_lag',
        'dest_outdegree_lag'
]

# COMMAND ----------

# 1. Add Repartitioning for Large Data
# num_partitions = 200  # Adjust based on cluster size (cores * 2-3)
# train_data = train_data.repartition(num_partitions)
# validation_data = validation_data.repartition(num_partitions//4)
# test_data = test_data.repartition(num_partitions//4)

# 2. Modified Pipeline with Debugging Checks
stages = []

# Process categorical features
for feature in categorical_features:
    indexer = StringIndexer(inputCol=feature, outputCol=f"{feature}_indexed", handleInvalid="keep")
    encoder = OneHotEncoder(inputCol=f"{feature}_indexed", outputCol=f"{feature}_encoded")
    stages += [indexer, encoder]

# Process high-cardinality features
for feature in high_cardinality_features:
    indexer = StringIndexer(inputCol=feature, outputCol=f"{feature}_numeric", handleInvalid="keep")
    stages.append(indexer)

# Add null check before assembling
stages.append(Imputer(inputCols=numerical_features, 
                    outputCols=numerical_features,
                    strategy="median"))

# Assemble features with debug
transformed_features = [f"{feature}_encoded" for feature in categorical_features] + \
                      [f"{feature}_numeric" for feature in high_cardinality_features] + \
                      numerical_features

assembler = VectorAssembler(inputCols=transformed_features, 
                          outputCol="features_unscaled", 
                          handleInvalid="keep")
stages.append(assembler)

# Add StandardScaler with mean centering disabled
scaler = StandardScaler(inputCol="features_unscaled", 
                      outputCol="features",
                      withStd=True, 
                      withMean=False)  # Critical for large datasets
stages.append(scaler)

# 3. Create Pipeline with Progress Checks
pipeline = Pipeline(stages=stages)

# 4. Fit Pipeline with Intermediate Validation
try:
    # Fit on sample data first
    sample_data = train_data.limit(1000)
    sample_model = pipeline.fit(sample_data)
    sample_transformed = sample_model.transform(sample_data)
    
    print("\n=== Sample Data Schema ===")
    sample_transformed.printSchema()
    
    print("\n=== Sample Features ===")
    # sample_transformed.select("features").show(5, truncate=False)
    
    # Now fit on full data
    pipeline_model = pipeline.fit(train_data)
    
except Exception as e:
    print(f"\n⚠️ Pipeline failed during fitting: {str(e)}")
    raise

# 5. Transform Data with Validation
def safe_transform(model, data, name):
    try:
        transformed = model.transform(data).cache()
        # Force execution and check features
        count = transformed.count()
        print(f"\n✅ Successfully transformed {name} dataset ({count} rows)")
        
        print(f"\n=== {name} Features Sample ===")
        # transformed.select("features").show(5, truncate=False)
        
        return transformed
    except Exception as e:
        print(f"\n⚠️ Transformation failed for {name}: {str(e)}")
        raise

train_data_transformed = safe_transform(pipeline_model, train_data, "Training")
validation_data_transformed = safe_transform(pipeline_model, validation_data, "Validation")
test_data_transformed = safe_transform(pipeline_model, test_data, "Test")

# COMMAND ----------

# Credit to 261 classmate Eric Mowat
from pyspark.ml.feature import OneHotEncoderModel
def get_one_hot_encoded_lengths(pipeline_model):
    """
    Get lengths of one-hot encoded columns from OneHotEncoderModel stages in the pipeline.

    Args:
        pipeline_model (PipelineModel): Trained pipeline model.

    Returns:
        dict: A dictionary mapping one-hot encoded column names to their lengths.
    """
    one_hot_lengths = {}
    for stage in pipeline_model.stages:
        if isinstance(stage, OneHotEncoderModel):
            input_col = stage.getInputCol()
            output_col = stage.getOutputCol()
            num_categories = stage.categorySizes[0]  # Number of categories for this column
            # If dropLast=True, subtract 1 from the number of categories
            adjusted_length = num_categories - (1 if stage.getDropLast() else 0)
            one_hot_lengths[output_col] = adjusted_length
           # print(f"OneHotEncoder detected. Input column: {input_col}, Output column: {output_col}, Categories: {adjusted_length}")
    return one_hot_lengths

# COMMAND ----------

# Print One Hot Encoder feature sizes
print("One Hot Encoded Feature sizes")
print(get_one_hot_encoded_lengths(pipeline_model))

# COMMAND ----------

print(train_data_transformed.columns)

# COMMAND ----------

display(train_data_transformed)

# COMMAND ----------

# Prepare data for modeling
train_data_ml = train_data_transformed.select("DEP_DEL15", "features_unscaled", "sample_weight")
validation_data_ml = validation_data_transformed.select("DEP_DEL15",  "features_unscaled")
test_data_ml = test_data_transformed.select("DEP_DEL15", "features_unscaled")

# COMMAND ----------

data_BASE_DIR = "dbfs:/student-groups/Group_04_04/"
display(dbutils.fs.ls(f"{data_BASE_DIR}"))

# COMMAND ----------

# Save the transformed encoded features
folder_path = f"dbfs:/student-groups/Group_04_04"

# Save train/validate/test split as a parquet files
# train_data_ml.write.parquet(f"{folder_path}/df_otpw_60M_train_data_encoded.parquet") # use to save a new version
# validation_data_ml.write.parquet(f"{folder_path}/df_otpw_60M_validation_data_encoded.parquet") # use to save a new version
# test_data_ml.write.parquet(f"{folder_path}/df_otpw_60M_test_data_encoded.parquet") # use to save a new version


train_data.write.mode('overwrite').parquet(f"{folder_path}/df_otpw_60M_train_data_encoded.parquet") # use if you want to overwrite exisiting file
validation_data.write.mode('overwrite').parquet(f"{folder_path}/df_otpw_60M_validation_data_encoded.parquet") # use if you want to overwrite exisiting file
test_data.write.mode('overwrite').parquet(f"{folder_path}/df_otpw_60M_test_data.parquet_encoded") # use if you want to overwrite exisiting file

# COMMAND ----------

# Evaluators for precision, recall, and F1 score

def evaluate_predictions(predictions, target_feature, dataset_name, f2_only=False):
    # Calculate confusion matrix components
    tp = predictions.filter((col(target_feature) == 1) & (col('prediction') == 1)).count()
    tn = predictions.filter((col(target_feature) == 0) & (col('prediction') == 0)).count()
    fp = predictions.filter((col(target_feature) == 0) & (col('prediction') == 1)).count()
    fn = predictions.filter((col(target_feature) == 1) & (col('prediction') == 0)).count()

    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    f2 = 5 * (precision * recall) / (4*precision + recall) if (precision + recall) != 0 else 0

    if (f2_only):
        return f2

    return {
        'dataset': dataset_name,
        'precision': precision,
        'recall': recall,
        'f2': f2,
        'confusion_matrix': [[tn, fp], [fn, tp]]
    }

# COMMAND ----------

def show_predictions_eval(prediction_datasets, target_feature):
    results = []
    for k,v in prediction_datasets.items():
        results.append(evaluate_predictions(v, target_feature, k))
    # Create metrics comparison table
    metrics_df = pd.DataFrame(results).set_index('dataset')
    print("\nMetrics Comparison:")
    print(metrics_df[['precision', 'recall', 'f2']].to_markdown(floatfmt=".3f"))

    # Plot confusion matrices
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for idx, result in enumerate(results):
        sns.heatmap(
            result['confusion_matrix'], 
            annot=True, fmt="d", 
            cmap="Blues",
            ax=axes[idx],
            cbar=False
        )
        axes[idx].set_title(f"{result['dataset']} Set\nConfusion Matrix")
        axes[idx].set_xlabel("Predicted")
        axes[idx].set_ylabel("Actual")

    plt.tight_layout()
    plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Baseline model - Logistic Regression

# COMMAND ----------

# Logistic Regression Model for classification
lr = LogisticRegression(
    featuresCol="features",
    labelCol="DEP_DEL15",
    weightCol="sample_weight",
    maxIter=10,
    regParam=0
)
lr_model = lr.fit(train_data_ml)

# Get predictions for all datasets
train_pred = lr_model.transform(train_data_ml)
val_pred = lr_model.transform(validation_data_ml)
test_pred = lr_model.transform(test_data_ml)
model_datasets = {'Training': train_pred, 'Validation': val_pred, 'Test': test_pred}
show_predictions_eval(model_datasets, "DEP_DEL15")

# COMMAND ----------

# Access the coefficients (weights)
coefficients = lr_model.coefficients

for coeff in coefficients:
    print(coeff)

# COMMAND ----------

len(coefficients)

# COMMAND ----------

# MAGIC %md
# MAGIC ### HyperTuning and Cross Validation For Logistic Regression

# COMMAND ----------

train_data_cv_total = train_data_transformed.withColumn("rank", percent_rank().over(Window.partitionBy().orderBy(["YEAR", "MONTH", "DAY_OF_MONTH", "CRS_DEP_TIME"])))
train_data_cv_total.cache()

# COMMAND ----------

# % of train_data_cv_total used as training
training_fold_percent_threshold = [0.2, 0.4, 0.6, 0.8]
regParams = [0.00001, 0.0002, 0.001, 0.01]
precision_cv = np.zeros(len(regParams))
recall_cv = np.zeros(len(regParams))
f1_cv = np.zeros(len(regParams))

# COMMAND ----------


# Didn't loop over tfpt for modularity in case of failure
tfpt=training_fold_percent_threshold[0]
train_data_cv = train_data_cv_total.filter(train_data_cv_total.rank <= tfpt)
val_data_cv = train_data_cv_total.filter(train_data_cv_total.rank > tfpt)
train_data_cv = train_data_cv.repartition(200) #Had a oom issue and was recommended to use repartition, but issue still remains. I ultimately restarted the cluster runtime and it works. So perhaps this is not needed
val_data_cv = val_data_cv.repartition(200)
for i, regParam in enumerate(regParams):
  lr_lasso = LogisticRegression(featuresCol="features", labelCol="DEP_DEL15",  weightCol="sample_weight", maxIter=10, regParam=regParam, elasticNetParam=1.0)
  lr_model_lasso = lr_lasso.fit(train_data_cv)
  lr_predictions = lr_model_lasso.transform(val_data_cv)

  eval_metrics = evaluate_predictions(lr_predictions, "DEP_DEL15", "Cross Validation")
  lr_precision_cv = eval_metrics["precision"]
  lr_recall_cv = eval_metrics["recall"]
  lr_f1_score_cv = eval_metrics["f1"]
  print(lr_precision_cv)
  print(lr_recall_cv)
  print(lr_f1_score_cv)
  precision_cv[i] += lr_precision_cv
  recall_cv[i] += lr_recall_cv
  f1_cv[i] += lr_f1_score_cv

# COMMAND ----------


# Didn't loop over tfpt for modularity in case of failure
tfpt=training_fold_percent_threshold[1]
train_data_cv = train_data_cv_total.filter(train_data_cv_total.rank <= tfpt)
val_data_cv = train_data_cv_total.filter(train_data_cv_total.rank > tfpt)
train_data_cv = train_data_cv.repartition(200)
val_data_cv = val_data_cv.repartition(200)
for i, regParam in enumerate(regParams):
  lr_lasso = LogisticRegression(featuresCol="features", labelCol="DEP_DEL15",  weightCol="sample_weight", maxIter=10, regParam=regParam, elasticNetParam=1.0)
  lr_model_lasso = lr_lasso.fit(train_data_cv)
  lr_predictions = lr_model_lasso.transform(val_data_cv)

  eval_metrics = evaluate_predictions(lr_predictions, "DEP_DEL15", "Cross Validation")
  lr_precision_cv = eval_metrics["precision"]
  lr_recall_cv = eval_metrics["recall"]
  lr_f1_score_cv = eval_metrics["f1"]
  print(lr_precision_cv)
  print(lr_recall_cv)
  print(lr_f1_score_cv)
  precision_cv[i] += lr_precision_cv
  recall_cv[i] += lr_recall_cv
  f1_cv[i] += lr_f1_score_cv

# COMMAND ----------


# Didn't loop over tfpt for modularity in case of failure
tfpt=training_fold_percent_threshold[2]
train_data_cv = train_data_cv_total.filter(train_data_cv_total.rank <= tfpt)
val_data_cv = train_data_cv_total.filter(train_data_cv_total.rank > tfpt)
train_data_cv = train_data_cv.repartition(200)
val_data_cv = val_data_cv.repartition(200)
for i, regParam in enumerate(regParams):
  lr_lasso = LogisticRegression(featuresCol="features", labelCol="DEP_DEL15",  weightCol="sample_weight", maxIter=10, regParam=regParam, elasticNetParam=1.0)
  lr_model_lasso = lr_lasso.fit(train_data_cv)
  lr_predictions = lr_model_lasso.transform(val_data_cv)

  eval_metrics = evaluate_predictions(lr_predictions, "DEP_DEL15", "Cross Validation")
  lr_precision_cv = eval_metrics["precision"]
  lr_recall_cv = eval_metrics["recall"]
  lr_f1_score_cv = eval_metrics["f1"]
  print(lr_precision_cv)
  print(lr_recall_cv)
  print(lr_f1_score_cv)
  precision_cv[i] += lr_precision_cv
  recall_cv[i] += lr_recall_cv
  f1_cv[i] += lr_f1_score_cv

# COMMAND ----------


# Didn't loop over tfpt for modularity in case of failure
tfpt=training_fold_percent_threshold[3]
train_data_cv = train_data_cv_total.filter(train_data_cv_total.rank <= tfpt)
val_data_cv = train_data_cv_total.filter(train_data_cv_total.rank > tfpt)
train_data_cv = train_data_cv.repartition(200)
val_data_cv = val_data_cv.repartition(200)
for i, regParam in enumerate(regParams):
  lr_lasso = LogisticRegression(featuresCol="features", labelCol="DEP_DEL15",  weightCol="sample_weight", maxIter=10, regParam=regParam, elasticNetParam=1.0)
  lr_model_lasso = lr_lasso.fit(train_data_cv)
  lr_predictions = lr_model_lasso.transform(val_data_cv)

  eval_metrics = evaluate_predictions(lr_predictions, "DEP_DEL15", "Cross Validation")
  lr_precision_cv = eval_metrics["precision"]
  lr_recall_cv = eval_metrics["recall"]
  lr_f1_score_cv = eval_metrics["f1"]
  print(lr_precision_cv)
  print(lr_recall_cv)
  print(lr_f1_score_cv)
  precision_cv[i] += lr_precision_cv
  recall_cv[i] += lr_recall_cv
  f1_cv[i] += lr_f1_score_cv

# COMMAND ----------

precision_cv = precision_cv / len(training_fold_percent_threshold)
recall_cv = recall_cv / len(training_fold_percent_threshold)
f1_cv = f1_cv / len(training_fold_percent_threshold)
print(precision_cv)
print("Best Precision under Cross Validation is " + str(np.max(precision_cv)) + "at regParam = " + str(regParams[np.argmax(precision_cv)]))
print("Best Recall under Cross Validation is " + str(np.max(recall_cv)) + "at regParam = " + str(regParams[np.argmax(recall_cv)]))
print("Best F1 under Cross Validation is " + str(np.max(f1_cv)) + "at regParam = " + str(regParams[np.argmax(f1_cv)]))

# COMMAND ----------

print(recall_cv)
print(f1_cv)

# COMMAND ----------

# MAGIC %md
# MAGIC ## XGBoost

# COMMAND ----------

# create a xgboost pyspark regressor estimator and set device="cuda"
xgboost_classifier = SparkXGBClassifier (
  features_col="features",
  label_col="DEP_DEL15",
  #device="cuda",
)

# train and return the model
xgboost_model = xgboost_classifier.fit(train_data_ml)

# COMMAND ----------

xgboost_train_predict_df = xgboost_model.transform(train_data_ml)
print(evaluate_predictions(xgboost_train_predict_df, "DEP_DEL15", "train"))
xgboost_val_predict_df = xgboost_model.transform(validation_data_ml)
print(evaluate_predictions(xgboost_val_predict_df, "DEP_DEL15", "validation"))
xgboost_test_predict_df = xgboost_model.transform(test_data_ml)
print(evaluate_predictions(xgboost_test_predict_df, "DEP_DEL15", "validation"))
model_datasets = {'Training': xgboost_train_predict_df, 'Validation': xgboost_val_predict_df, 'Test': xgboost_test_predict_df}
show_predictions_eval(model_datasets, "DEP_DEL15")

# COMMAND ----------

booster = xgboost_model.get_booster()
booster.get_score(importance_type='gain')

# COMMAND ----------

# Use the raw data for temporal ranking, but transform it with the pipeline
train_data_cv_total = pipeline_model.transform(
    train_data.withColumn(
        "rank", 
        percent_rank().over(Window.partitionBy().orderBy(["YEAR", "MONTH", "DAY_OF_MONTH"]))
    )
).repartition(20).cache()  # Now contains both raw columns and encoded features
training_fold_percent_threshold = [0.2, 0.4, 0.6, 0.8]
rolling_time_series_cv_train = {}
rolling_time_series_cv_val = {}
for tfpt in training_fold_percent_threshold:
    rolling_time_series_cv_train[tfpt] = train_data_cv_total.filter(train_data_cv_total.rank <= tfpt).cache()
    rolling_time_series_cv_val[tfpt] = train_data_cv_total.filter(train_data_cv_total.rank > tfpt).cache()

# COMMAND ----------

import optuna

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 150),
        "max_depth": trial.suggest_int("max_depth", 5, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.2, 1, log=True),
    }
    
    avg_score = 0
    for tfpt in training_fold_percent_threshold:
        train_data_cv = rolling_time_series_cv_train[tfpt]
        val_data_cv = rolling_time_series_cv_val[tfpt]

        xgb = SparkXGBClassifier(
            features_col="features_unscaled",
            label_col="DEP_DEL15",
            #device="cuda",
            **params
        )
        model = xgb.fit(train_data_cv)
        predictions = model.transform(val_data_cv)
        score = evaluate_predictions(predictions, "DEP_DEL15", "cv", f2_only=True)
        avg_score += score

    return avg_score / len(training_fold_percent_threshold)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30, timeout=3600)

print("Best trial:", study.best_trial.params)

# COMMAND ----------

train_data_ml = train_data_ml.repartition(160)

# COMMAND ----------

# With Hyperparameter Tuning
xgboost_classifier = SparkXGBClassifier (
  features_col="features",
  label_col="DEP_DEL15",
  n_estimators=115,
  max_depth=9,
  learning_rate=0.27122395210228795,
  #device="cuda",
)

# train and return the model
xgboost_model = xgboost_classifier.fit(train_data_ml)

# COMMAND ----------

xgboost_train_predict_df = xgboost_model.transform(train_data_ml)
print(evaluate_predictions(xgboost_train_predict_df, "DEP_DEL15", "train"))
xgboost_val_predict_df = xgboost_model.transform(validation_data_ml)
print(evaluate_predictions(xgboost_val_predict_df, "DEP_DEL15", "validation"))
xgboost_test_predict_df = xgboost_model.transform(test_data_ml)
print(evaluate_predictions(xgboost_test_predict_df, "DEP_DEL15", "validation"))
model_datasets = {'Training': xgboost_train_predict_df, 'Validation': xgboost_val_predict_df, 'Test': xgboost_test_predict_df}
show_predictions_eval(model_datasets, "DEP_DEL15")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Neural Networks with Tensorflow

# COMMAND ----------

# Add sample_weight and label BEFORE flattening
train_data_trans = train_data_transformed \
    .withColumn("label", col("DEP_DEL15")) \
    .withColumn("sample_weight", col("sample_weight"))

# Use the raw data for temporal ranking
train_data_cv_total = train_data_trans.withColumn(
    "rank", 
    percent_rank().over(Window.partitionBy().orderBy(["YEAR", "MONTH", "DAY_OF_MONTH"]))
)

# Now convert features to array
train_flat = train_data_cv_total.withColumn("features_array", vector_to_array("features_unscaled"))
feature_length = len(train_flat.select("features_array").first()[0])
feature_cols = [f"f_{i}" for i in range(feature_length)]

# Select feature columns + label + weight + rank
train_flat = train_flat.select(
    *[col("features_array")[i].alias(feature_cols[i]) for i in range(feature_length)],
    "label",
    "sample_weight",
    "rank"
).cache()

# Same for val/test (without time series ranking)
val_data_trans = validation_data_transformed.withColumn("label", col("DEP_DEL15"))
val_flat = val_data_trans.withColumn("features_array", vector_to_array("features_unscaled"))
val_flat = val_flat.select(
    *[col("features_array")[i].alias(feature_cols[i]) for i in range(feature_length)],
    "label"
).cache()

test_data_trans = test_data_transformed.withColumn("label", col("DEP_DEL15"))
test_flat = test_data_trans.withColumn("features_array", vector_to_array("features_unscaled"))
test_flat = test_flat.select(
    *[col("features_array")[i].alias(feature_cols[i]) for i in range(feature_length)],
    "label"
).cache()

# COMMAND ----------

folder_path = f"dbfs:/student-groups/Group_04_04"

# Save train/validate/test split as a parquet files
# train_flat.write.parquet(f"{folder_path}/df_otpw_60M_train_flat_petastorm.parquet") # use to save a new version
# val_flat.write.parquet(f"{folder_path}/df_otpw_60M_validation_flat_petastorm.parquet") # use to save a new version
# test_flat.write.parquet(f"{folder_path}/df_otpw_60M_test_flat_petastorm.parquet") # use to save a new version


# Save train/validate/test split as parquet files
train_flat.write.mode("overwrite").parquet(f"{folder_path}/df_otpw_2015_2024_train_flat.parquet")
val_flat.write.mode("overwrite").parquet(f"{folder_path}/df_otpw_2015_2024_validation_flat.parquet")
test_flat.write.mode("overwrite").parquet(f"{folder_path}/df_otpw_2015_2024_test_flat.parquet")

# COMMAND ----------

# folder_path = f"dbfs:/student-groups/Group_04_04"

# Enable Arrow-based conversion
spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")

# Load data
train_df = spark.read.parquet("dbfs:/student-groups/Group_04_04/df_otpw_2015_2024_train_flat.parquet").toPandas()
val_df = spark.read.parquet("dbfs:/student-groups/Group_04_04/df_otpw_2015_2024_validation_flat.parquet").toPandas()
test_df = spark.read.parquet("dbfs:/student-groups/Group_04_04/df_otpw_2015_2024_test_flat.parquet").toPandas()

# COMMAND ----------

# Feature columns
feature_cols = [c for c in train_df.columns if c.startswith("f_")]

# Prepare arrays for validation and test sets
X_val = val_df[feature_cols].values
y_val = val_df["label"].values

X_test = test_df[feature_cols].values
y_test = test_df["label"].values

# Create time series CV splits
training_fold_percent_threshold = [0.2, 0.4, 0.6, 0.8]
cv_train_dfs = {}
cv_val_dfs = {}

for tfpt in training_fold_percent_threshold:
    cv_train_dfs[tfpt] = train_df[train_df['rank'] <= tfpt].copy()
    cv_val_dfs[tfpt] = train_df[train_df['rank'] > tfpt].copy()

# COMMAND ----------

# Model Configuration
def create_model(hidden_layers=[128], dropout_rate=0.3, l2_reg=0.001, use_batch_norm=False):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(len(feature_cols),)))
    
    for units in hidden_layers:
        model.add(tf.keras.layers.Dense(
            units, 
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(l2_reg)
        ))
        if use_batch_norm:
            model.add(tf.keras.layers.BatchNormalization())

        model.add(tf.keras.layers.Dropout(dropout_rate))
    
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    return model

# COMMAND ----------

# Training Function with Your Exact Setup
def train_model(model, X_train_data, y_train_data, X_val_data, y_val_data, 
                sample_weight_data=None, batch_size=256, epochs=100, lr=0.0005, verbose=0):
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, 
                loss='binary_crossentropy', 
                metrics=['accuracy', tf.keras.metrics.Recall(), tf.keras.metrics.AUC()])
    
    # Add learning rate scheduler
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=0
    )
    
    history = model.fit(
        X_train_data, y_train_data,
        sample_weight=sample_weight_data,
        validation_data=(X_val_data, y_val_data),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', 
                patience=5,
                restore_best_weights=True
            ),
            lr_scheduler
        ],
        verbose=verbose
    )
    return history

# COMMAND ----------

# Evaluation Functions
def evaluate_model(model, X, y):
    pred_proba = model.predict(X, verbose=0)
    pred_labels = (pred_proba >= 0.5).astype(int).flatten()
    results_df = pd.DataFrame({
        "label": y,
        "prediction": pred_labels,
        "probability": pred_proba.flatten()
    })
    spark_df = spark.createDataFrame(results_df)
    return evaluate_predictions(spark_df, "label", "dataset_name")

def plot_confusion_matrix_grid(grid_results, target_col):
    """Plot confusion matrices for all configurations"""
    n_models = len(grid_results)
    
    # Check if there are any results to plot
    if n_models == 0:
        print("No models to visualize.")
        return
    
    fig, axes = plt.subplots(max(1, n_models), 3, figsize=(18, 5*max(1, n_models)))
    
    # Handle the case where there's only one model
    if n_models == 1:
        axes = [axes]
    
    for i, result in enumerate(grid_results):
        params = result['params']
        title_suffix = f"\nLayers: {params.get('hidden_layers', [])} " \
                      f"| BS: {params.get('batch_size', '')} " \
                      f"| LR: {params.get('learning_rate', '')} " \
                      f"| DO: {params.get('dropout_rate', '')}"
        
        for j, dataset in enumerate(['train', 'val', 'test']):
            cm = np.array(result['metrics'][dataset]['confusion_matrix'])
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                      ax=axes[i][j],
                      cbar=False)
            axes[i][j].set_title(f"Config {i+1} {dataset.capitalize()}{title_suffix}")
            axes[i][j].set_xlabel("Predicted")
            axes[i][j].set_ylabel("Actual")
    
    plt.tight_layout()
    plt.show()

def compare_results(grid_results):
    """Enhanced Comparison Table"""
    # Check if there are any results to compare
    if not grid_results:
        print("No models to compare.")
        return pd.DataFrame()
        
    comparison_data = []
    
    for result in grid_results:
        params = result['params']
        metrics = result['metrics']
        
        comparison_data.append({
            'Layers': str(params.get('hidden_layers', [128])),
            'Batch Size': params.get('batch_size', 256),
            'LR': params.get('learning_rate', 0.0005),
            'Dropout': params.get('dropout_rate', 0.3),
            'L2': params.get('l2_reg', 0.001),
            'BatchNorm': params.get('use_batch_norm', False),
            'CV F2': metrics['cv']['f2'] if 'cv' in metrics else metrics['train']['f2'],
            'CV Recall': metrics['cv']['recall'] if 'cv' in metrics else metrics['train']['recall'],
            'Val F2': metrics['val']['f2'],
            'Val Recall': metrics['val']['recall'],
            'Test F2': metrics['test']['f2'],
            'Test Recall': metrics['test']['recall']
        })
    
    df = pd.DataFrame(comparison_data).sort_values('CV F2', ascending=False)
    
    print("\nFinal Comparison:")
    print(df.to_markdown(
        index=False, 
        floatfmt=".4f",
        headers=['Layers', 'Batch', 'LR', 'DO', 'L2', 'BN',
                'CV-F2', 'CV-Rcl', 'ValF2', 'ValRcl',
                'TstF2', 'TstRcl']
    ))
    
    return df

# COMMAND ----------

def run_optuna_trial_and_get_best_params(n_trials=1):
    study = optuna.create_study(
        direction='maximize',
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10, interval_steps=1)
    )

    def objective(trial):
        n_layers = trial.suggest_int('n_layers', 1, 3)
        hidden_layers = [trial.suggest_int(f'units_l{i}', 32, 512, step=32) for i in range(n_layers)]
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
        l2_reg = trial.suggest_float('l2_reg', 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [128, 256, 512, 1024])
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        use_batch_norm = trial.suggest_categorical('use_batch_norm', [True, False])

        cv_f2_scores = []
        for tfpt in training_fold_percent_threshold:
            train_fold_df = cv_train_dfs[tfpt]
            val_fold_df = cv_val_dfs[tfpt]
            X_train_fold = train_fold_df[feature_cols].values
            y_train_fold = train_fold_df["label"].values
            sample_weight_fold = train_fold_df["sample_weight"].values if "sample_weight" in train_fold_df.columns else None
            X_val_fold = val_fold_df[feature_cols].values
            y_val_fold = val_fold_df["label"].values

            model = create_model(hidden_layers, dropout_rate, l2_reg, use_batch_norm)
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                loss='binary_crossentropy',
                metrics=['accuracy', tf.keras.metrics.Recall(), tf.keras.metrics.AUC()]
            )

            model.fit(
                X_train_fold, y_train_fold,
                sample_weight=sample_weight_fold,
                validation_data=(X_val_fold, y_val_fold),
                epochs=20,
                batch_size=batch_size,
                callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)],
                verbose=0
            )

            pred_proba = model.predict(X_val_fold, verbose=0)
            pred_labels = (pred_proba >= 0.5).astype(int).flatten()
            results_df = pd.DataFrame({
                "label": y_val_fold,
                "prediction": pred_labels,
                "probability": pred_proba.flatten()
            })
            spark_df = spark.createDataFrame(results_df)
            fold_metrics = evaluate_predictions(spark_df, "label", f"cv_{tfpt}", f2_only=True)
            cv_f2_scores.append(fold_metrics)

        return np.mean(cv_f2_scores)

    try:
        study.optimize(objective, n_trials=n_trials)
        return study.best_params, study.best_trial
    except Exception as e:
        print(f"Optuna error: {e}")
        return None, None

# COMMAND ----------

def train_and_evaluate_final_model(best_params):
    # Extract best hyperparameters
    hidden_layers = [best_params[f'units_l{i}'] for i in range(best_params['n_layers'])]
    dropout_rate = best_params['dropout_rate']
    l2_reg = best_params['l2_reg']
    batch_size = best_params['batch_size']
    learning_rate = best_params['learning_rate']
    use_batch_norm = best_params['use_batch_norm']

    # Prepare training data
    X_train = train_df[feature_cols].values
    y_train = train_df["label"].values
    sample_weight = train_df["sample_weight"].values if "sample_weight" in train_df.columns else None

    # Build final model
    model = create_model(hidden_layers, dropout_rate, l2_reg, use_batch_norm)
    history = train_model(
        model, X_train, y_train, X_val, y_val,
        sample_weight, batch_size, 100, learning_rate, verbose=1
    )

    # Evaluate on train, val, test
    metrics = {
        'train': evaluate_model(model, X_train, y_train),
        'val': evaluate_model(model, X_val, y_val),
        'test': evaluate_model(model, X_test, y_test)
    }

    # model.save("best_flight_delay_model.keras")
    return model, history, metrics


# COMMAND ----------

# MAGIC %md
# MAGIC ### Experiment 1 - Provide Manual Grid search for Data between 2015-2021 ( Train : 2015-2019; Validation: 2020, Test: 2021)

# COMMAND ----------

param_grid = [
    {'hidden_layers': [128], 'batch_size': 256, 
     'learning_rate': 0.0005, 'dropout_rate': 0.2},
    {'hidden_layers': [128, 64], 'batch_size': 512,
     'learning_rate': 0.0003, 'dropout_rate': 0.3},
    {'hidden_layers': [256, 128, 64], 'batch_size': 1024,
     'learning_rate': 0.0001, 'dropout_rate': 0.5},
    {'hidden_layers': [128], 'batch_size': 256,
     'learning_rate': 0.001, 'dropout_rate': 0.4}
]
    
# Run grid search
grid_results = run_grid_search(param_grid)

# Show comparison
compare_results(grid_results)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Experiment 2 - Find Best Hyperparameters using Optuna and Cross Validation for Data between 2015-2024 ( Train : 2015-2021; Validation: 2022-2023, Test: 2024)

# COMMAND ----------

print("🔍 Running one Optuna trial to get best parameters...")
best_params, best_trial = run_optuna_trial_and_get_best_params(n_trials=1)

print("\n Best hyperparameters:")
for k, v in best_params.items():
    print(f"{k}: {v}")

# COMMAND ----------

# These parameters have been extracted from the best trial run conducted above, due to the randomness of Optuna, the console log above might now show different values
best_params = {
    'n_layers': 1,
    'units_l0': 448,
    'dropout_rate': 0.27565986839411283,
    'l2_reg': 0.0001554892275405439,
    'batch_size': 256,
    'learning_rate': 1.1258097381851819e-05,
    'use_batch_norm': False
}

# COMMAND ----------

train_df[feature_cols] = train_df[feature_cols].astype(np.float32)
val_df[feature_cols] = val_df[feature_cols].astype(np.float32)
test_df[feature_cols] = test_df[feature_cols].astype(np.float32)

# COMMAND ----------

print("\nBest hyperparameters:")
for k, v in best_params.items():
    print(f"{k}: {v}")

print("\nTraining final model with best hyperparameters...")
final_model, final_history, final_metrics = train_and_evaluate_final_model(best_params)

print("\nFinal Evaluation:")
for split in ['train', 'val', 'test']:
    print(f"{split.upper()} F2: {final_metrics[split]['f2']:.4f} | Recall: {final_metrics[split]['recall']:.4f}")


# COMMAND ----------

for split in ['train', 'val', 'test']:
    m = final_metrics[split]
    print(f"{split.upper()} F2: {m['f2']:.4f} | Recall: {m['recall']:.4f} | Precision: {m['precision']:.4f}")

# COMMAND ----------

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, split in enumerate(['train', 'val', 'test']):
    cm = np.array(final_metrics[split]['confusion_matrix'])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[idx])
    axes[idx].set_title(f"{split.capitalize()} Confusion Matrix")
    axes[idx].set_xlabel("Predicted")
    axes[idx].set_ylabel("Actual")

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## BiLSTM on the hyperparameters found above

# COMMAND ----------

def create_bilstm_model(units=128, dropout_rate=0.3, l2_reg=0.001):
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(1, len(feature_cols))),
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(units, return_sequences=False,
                                 kernel_regularizer=tf.keras.regularizers.l2(l2_reg))
        ),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])


# COMMAND ----------

def train_and_evaluate_bilstm_model(best_params):
    # Extract best hyperparameters
    units = best_params['units_l0']  # Only one layer for LSTM
    dropout_rate = best_params['dropout_rate']
    l2_reg = best_params['l2_reg']
    batch_size = best_params['batch_size']
    learning_rate = best_params['learning_rate']

    # Prepare training data with sequence dimension
    X_train = train_df[feature_cols].values.astype(np.float32).reshape((-1, 1, len(feature_cols)))
    y_train = train_df["label"].values
    sample_weight = train_df["sample_weight"].values if "sample_weight" in train_df.columns else None

    X_val_seq = X_val.reshape((-1, 1, X_val.shape[1]))
    X_test_seq = X_test.reshape((-1, 1, X_test.shape[1]))

    # Build LSTM model
    model = create_bilstm_model(units, dropout_rate, l2_reg)

    history = train_model(
        model, X_train, y_train, X_val_seq, y_val,
        sample_weight, batch_size, 100, learning_rate, verbose=1
    )

    # Evaluate
    metrics = {
        'train': evaluate_model(model, X_train, y_train),
        'val': evaluate_model(model, X_val_seq, y_val),
        'test': evaluate_model(model, X_test_seq, y_test)
    }

    return model, history, metrics

# COMMAND ----------

bilstm_model, bilstm_history, bilstm_metrics = train_and_evaluate_bilstm_model(best_params)

# COMMAND ----------

for split in ['train', 'val', 'test']:
    m = bilstm_metrics[split]
    print(f"{split.upper()} F2: {m['f2']:.4f} | Recall: {m['recall']:.4f} | Precision: {m['precision']:.4f}")

# COMMAND ----------

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, split in enumerate(['train', 'val', 'test']):
    cm = np.array(bilstm_metrics[split]['confusion_matrix'])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[idx])
    axes[idx].set_title(f"{split.capitalize()} Confusion Matrix")
    axes[idx].set_xlabel("Predicted")
    axes[idx].set_ylabel("Actual")

plt.tight_layout()
plt.show()