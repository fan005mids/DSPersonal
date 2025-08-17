# Databricks notebook source
%pip install folium
%pip install tabulate

# COMMAND ----------

from pyspark.sql.functions import col, sum as spark_sum, count, desc, asc, year,when, isnan, avg, min, max, hour, round
from pyspark.sql.types import DoubleType, FloatType, StringType, IntegerType, DecimalType
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import functions as F
from pyspark.sql.window import Window
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from folium.plugins import MarkerCluster
import calendar
import branca.colormap as cm
from folium.plugins import MiniMap
import matplotlib.cm as cm
import pyspark.pandas as ps
from branca.colormap import LinearColormap
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, RegressionEvaluator, MulticlassClassificationEvaluator
from pyspark.sql.functions import percent_rank
from pyspark.sql import Window
from pyspark.ml.stat import Correlation
print("Welcome to the W261 final project!") 

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
df_flights_WN = spark.read.parquet("dbfs:/mnt/mids-w261/datasets_final_project_2022/parquet_airlines_data/")

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

# Convert Spark DataFrame to Pandas for visualization
df_WN_pd = df_WN_yearly_delays.toPandas()

# Compute mean delay percentage across all years
mean_delay_percentage = df_WN_pd["Delay_Percentage_WN"].mean()

# Create figure
plt.figure(figsize=(10, 6))

# Plot bar chart
bars = plt.bar(df_WN_pd["Year"], df_WN_pd["Delay_Percentage_WN"], color="skyblue", label="Yearly Delay Percentage")

# Add a horizontal mean line
plt.axhline(y=mean_delay_percentage, color='red', linestyle='--', linewidth=1, label=f"Mean Delay Percentage ({mean_delay_percentage:.2f}%)")

# Add text labels slightly lower for better visibility
for bar in bars:
    yval = bar.get_height()  # Get height of bar (delay percentage)
    plt.text(bar.get_x() + bar.get_width()/2, yval - 1, f"{yval:.2f}%", ha='center', fontsize=10)

# Customize chart
plt.xlabel("Year")
plt.ylabel("Delay Percentage (%)")
plt.title("Southwest Airlines Year-over-Year Flight Delay Trend (2015-2021)")
plt.xticks(df_WN_pd["Year"])  # Ensure all years appear on x-axis
plt.legend()

# **REMOVE the grey dotted lines**
plt.grid(False)  # Disables the grid completely

# **Save the plot as an image**
chart_path = "/dbfs/FileStore/WN_flight_delays_1.png"  # Path in DBFS
plt.savefig(chart_path, dpi=300, bbox_inches='tight')  # High-quality PNG file

# Show plot
plt.show()

# Print path for reference
print(f"Chart saved at: {chart_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Query for Custom Join

# COMMAND ----------

df_flights = spark.read.parquet(f"dbfs:/mnt/mids-w261/datasets_final_project_2022/parquet_airlines_data_1y")
df_weather = spark.read.parquet(f"dbfs:/mnt/mids-w261/datasets_final_project_2022/parquet_weather_data_1y/")
df_stations = spark.read.parquet(f"dbfs:/mnt/mids-w261/datasets_final_project_2022/stations_data/stations_with_neighbors.parquet/")
df_airport_codes = spark.read.format("csv").option("header","true").load(f"dbfs:/mnt/mids-w261/airport-codes_csv.csv")

df_flights.createOrReplaceTempView("flights")
df_weather.createOrReplaceTempView("weather")
df_stations.createOrReplaceTempView("stations")
df_airport_codes.createOrReplaceTempView("airport_codes")

df_weather2 = spark.sql("""
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
        weather
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
            DISTINCT neighbor_id,
            neighbor_call,
            neighbor_name,
            neighbor_lat,
            neighbor_lon
        FROM
            stations
    ),
    distances AS (
        SELECT
            ac.iata_code,
            ac.name,
            ac.type,
            ac.latitude,
            ac.longitude,
            s.neighbor_id,
            s.neighbor_call,
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
    )
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
        flights f
        LEFT JOIN state_time_zones st
            ON f.ORIGIN_STATE_ABR = st.state
        LEFT JOIN nearest_neighbor nn
            ON f.ORIGIN = nn.iata_code
        LEFT JOIN nearest_neighbor nn2
            ON f.DEST = nn.iata_code
""")

df_flights2 = df_flights2.withColumn("join_date", F.date_trunc("hour", F.col("four_hours_prior_depart_UTC")))
df_weather2 = df_weather2.withColumn("join_date", F.date_trunc("hour", F.col("DATE")))

df_joined = (df_flights2
    .join(df_weather2,
        (F.col("origin_station_id") == F.col("STATION")) &
        (df_flights2.join_date == df_weather2.join_date) &
        (F.col("DATE").between(
            F.col("four_hours_prior_depart_UTC"),
            F.col("two_hours_prior_depart_UTC")
        ))
    )
    .drop("join_date")
)

df_joined.write.mode("overwrite").parquet(f"dbfs:/student-groups/Group_04_04/df_joined_1y.parquet")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Exploratory Data Analysis

# COMMAND ----------

# OTPW
df_otpw = spark.read.csv("dbfs:/mnt/mids-w261/OTPW_12M/OTPW_12M/OTPW_12M_2015.csv.gz", header=True, inferSchema=True)
display(df_otpw)

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
num_rows = df_otpw.count()
num_cols = len(df_otpw.columns)

print(f"✅ Dataset contains {num_rows:,} rows and {num_cols} columns.")

null_percentage_df = calculate_null_percentage(df_otpw)
display(null_percentage_df)

# COMMAND ----------

# List of selected features
selected_features = [
    "DEP_DEL15", "FL_DATE", "YEAR", "QUARTER", "MONTH", 
    "DAY_OF_WEEK", "DAY_OF_MONTH", "OP_CARRIER", "ORIGIN", "DEST", "ORIGIN_CITY_NAME", "DEST_CITY_NAME", "CRS_DEP_TIME", "CRS_ARR_TIME", "ORIGIN_STATE_ABR", "DEST_STATE_ABR", "origin_type",
    "dest_type", "origin_airport_lat", "origin_airport_lon", "dest_airport_lat", "dest_airport_lon", "DISTANCE", "DISTANCE_GROUP", "CANCELLED", "CANCELLATION_CODE",
    "DIVERTED", "CARRIER_DELAY", "WEATHER_DELAY", "NAS_DELAY", "SECURITY_DELAY", "LATE_AIRCRAFT_DELAY",
    "HourlyDewPointTemperature", "HourlyDryBulbTemperature", "HourlyPrecipitation", "HourlyRelativeHumidity", "HourlySkyConditions", "HourlySeaLevelPressure",
    "HourlyVisibility", "HourlyWindDirection", "HourlyWindSpeed", 
]

# Select only required columns
df_selected = df_otpw.select(*selected_features)

# Filter for Southwest Airlines only
df_southwest = df_selected.filter(col("OP_CARRIER") == "WN")

num_rows = df_southwest.count()
num_cols = len(df_southwest.columns)

print(f"✅ Dataset after filtering contains {num_rows:,} rows and {num_cols} columns.")

# COMMAND ----------

summary_stats = df_southwest.describe()
display(summary_stats)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Basic cleaning of selected data to help with visualization

# COMMAND ----------

def convert_column_types(df):
    """
    Function to convert column data types appropriately for the flight delay dataset.
    
    Args:
        df: PySpark DataFrame to convert
        
    Returns:
        PySpark DataFrame with converted column types
    """
    # Define column type mappings
    string_columns = [
        "FL_DATE", "OP_CARRIER", "ORIGIN", "DEST", "ORIGIN_CITY_NAME", 
        "DEST_CITY_NAME", "ORIGIN_STATE_ABR", "DEST_STATE_ABR", "origin_type",
        "dest_type", "CANCELLATION_CODE", "HourlySkyConditions"
    ]
    
    double_columns = [
        "DEP_DEL15", "YEAR", "QUARTER", "MONTH", "DAY_OF_WEEK", "DAY_OF_MONTH",
        "CRS_DEP_TIME", "CRS_ARR_TIME", "origin_airport_lat", "origin_airport_lon", 
        "dest_airport_lat", "dest_airport_lon", "DISTANCE", "DISTANCE_GROUP", 
        "CANCELLED", "DIVERTED", "CARRIER_DELAY", "WEATHER_DELAY", "NAS_DELAY", 
        "SECURITY_DELAY", "LATE_AIRCRAFT_DELAY"
    ]
    
    decimal_columns = [
        "HourlyDewPointTemperature", "HourlyDryBulbTemperature", "HourlyPrecipitation",
        "HourlyRelativeHumidity", "HourlySeaLevelPressure", "HourlyVisibility",
        "HourlyWindDirection", "HourlyWindSpeed"
    ]
    
    # Convert string columns
    for column in string_columns:
        if column in df.columns:
            df = df.withColumn(column, col(column).cast(StringType()))
    
    # Convert double columns
    for column in double_columns:
        if column in df.columns:
            df = df.withColumn(column, col(column).cast(DoubleType()))
    
    # Convert decimal columns (with higher precision for weather data)
    for column in decimal_columns:
        if column in df.columns:
            df = df.withColumn(column, col(column).cast(DecimalType(10, 4)))
    
    return df

# COMMAND ----------

def clean_flight_data(df):

    # Handle DEP_DEL15 for cancelled flights - assign value 2 to represent "cancelled"
    df = df.withColumn(
        "DEP_DEL15", 
        F.when(F.col("DEP_DEL15").isNull() & (F.col("CANCELLED") == 1), 2).otherwise(F.col("DEP_DEL15"))
    )

    # Drop rows with missing values in critical columns EXCEPT when the flight is cancelled
    critical_cols = ["FL_DATE", "OP_CARRIER", "ORIGIN", "DEST", "CRS_DEP_TIME", "CRS_ARR_TIME"]
    for col in critical_cols:
        df = df.filter((F.col(col).isNotNull()) | (F.col("CANCELLED") == 1))

    # Extract temporal features from FL_DATE if missing
    df = df.withColumn("YEAR_TEMP", F.year("FL_DATE"))
    df = df.withColumn("QUARTER_TEMP", F.quarter("FL_DATE"))
    df = df.withColumn("MONTH_TEMP", F.month("FL_DATE"))
    df = df.withColumn("DAY_OF_WEEK_TEMP", F.dayofweek("FL_DATE"))
    df = df.withColumn("DAY_OF_MONTH_TEMP", F.dayofmonth("FL_DATE"))

    # Replace missing values with extracted values
    df = df.withColumn("YEAR", F.coalesce(F.col("YEAR"), F.col("YEAR_TEMP")))
    df = df.withColumn("QUARTER", F.coalesce(F.col("QUARTER"), F.col("QUARTER_TEMP")))
    df = df.withColumn("MONTH", F.coalesce(F.col("MONTH"), F.col("MONTH_TEMP")))
    df = df.withColumn("DAY_OF_WEEK", F.coalesce(F.col("DAY_OF_WEEK"), F.col("DAY_OF_WEEK_TEMP")))
    df = df.withColumn("DAY_OF_MONTH", F.coalesce(F.col("DAY_OF_MONTH"), F.col("DAY_OF_MONTH_TEMP")))

    # Drop temporary columns
    df = df.drop("YEAR_TEMP", "QUARTER_TEMP", "MONTH_TEMP", "DAY_OF_WEEK_TEMP", "DAY_OF_MONTH_TEMP")

    # Rolling Mean Imputation for weather metrics with small gaps
    window_spec = Window.partitionBy("ORIGIN").orderBy("FL_DATE").rowsBetween(-5, 5) #ToDo: also add depature time here

    weather_cols_rolling = ["HourlyWindSpeed", "HourlyWindDirection", "HourlyDewPointTemperature", 
                            "HourlyDryBulbTemperature", "HourlyVisibility", "HourlyRelativeHumidity"]

    for col in weather_cols_rolling:
        df = df.withColumn(f"{col}_rolling_mean", F.avg(F.col(col)).over(window_spec))
        df = df.withColumn(col, F.coalesce(F.col(col), F.col(f"{col}_rolling_mean")))
        df = df.drop(f"{col}_rolling_mean")

    # Airport/Month-Specific Median Values for precipitation and sea level pressure
    weather_cols_median = ["HourlyPrecipitation", "HourlySeaLevelPressure"]

    for col in weather_cols_median:
        median_expr = F.expr(f"percentile_approx({col}, 0.5)")
        medians = df.groupBy("ORIGIN", "MONTH").agg(median_expr.alias(f"{col}_median"))
        df = df.join(medians, on=["ORIGIN", "MONTH"], how="left")
        df = df.withColumn(col, F.coalesce(F.col(col), F.col(f"{col}_median")))
        df = df.drop(f"{col}_median")

    # Mode Imputation for HourlySkyConditions
    sky_conditions_mode = df.groupBy("ORIGIN", "MONTH").agg(F.mode("HourlySkyConditions").alias("sky_conditions_mode"))
    df = df.join(sky_conditions_mode, on=["ORIGIN", "MONTH"], how="left")
    df = df.withColumn("HourlySkyConditions", F.coalesce(F.col("HourlySkyConditions"), F.col("sky_conditions_mode")))
    df = df.drop("sky_conditions_mode")

    # Handle categorical columns with 'UNKNOWN' for outliers
    categorical_cols = ["ORIGIN_STATE_ABR", "DEST_STATE_ABR", "origin_type", "dest_type"]
    for col in categorical_cols:
        df = df.withColumn(col, F.when(F.col(col).isNull(), "UNKNOWN").otherwise(F.col(col)))

    # Handle distance and distance group for outliers
    df = df.withColumn("DISTANCE", F.when(F.col("DISTANCE") < 0, None).otherwise(F.col("DISTANCE")))
    df = df.withColumn("DISTANCE_GROUP", F.when(F.col("DISTANCE_GROUP") < 0, None).otherwise(F.col("DISTANCE_GROUP")))

    # Impute missing values for distance and distance group with median
    distance_median = df.select(F.expr("percentile_approx(DISTANCE, 0.5)").alias("distance_median")).first()["distance_median"]
    distance_group_median = df.select(F.expr("percentile_approx(DISTANCE_GROUP, 0.5)").alias("dg_median")).first()["dg_median"]
    df = df.withColumn("DISTANCE", F.coalesce(F.col("DISTANCE"), F.lit(distance_median)))
    df = df.withColumn("DISTANCE_GROUP", F.coalesce(F.col("DISTANCE_GROUP"), F.lit(distance_group_median)))

    # For any remaining nulls in geographic coordinates, use airport-specific values
    for col in ["origin_airport_lat", "origin_airport_lon", "dest_airport_lat", "dest_airport_lon"]:
        airport_col = "ORIGIN" if "origin" in col else "DEST"
        common_values = df.filter(F.col(col).isNotNull()).groupBy(airport_col).agg(F.first(col).alias(f"{col}_common"))
        df = df.join(common_values, on=airport_col, how="left")
        df = df.withColumn(col, F.coalesce(F.col(col), F.col(f"{col}_common")))
        df = df.drop(f"{col}_common")

    # After all imputations, check if any nulls remain in weather columns and use global medians as last resort
    all_weather_cols = weather_cols_rolling + weather_cols_median + ["HourlySkyConditions"]
    for col in all_weather_cols:
        global_median = df.select(F.expr(f"percentile_approx({col}, 0.5)").alias("global_median")).first()["global_median"]
        df = df.withColumn(col, F.coalesce(F.col(col), F.lit(global_median)))

    return df

# COMMAND ----------

df_southwest_type_conversion = convert_column_types(df_southwest)

# COMMAND ----------

df_southwest_cleaned = clean_flight_data(df_southwest_type_conversion)

# COMMAND ----------

# Analyze missing values in cleaned features
null_percentage_df = calculate_null_percentage(df_southwest_cleaned)
display(null_percentage_df)

# COMMAND ----------

display(df_southwest_cleaned)

# COMMAND ----------

# Create folder
# section = "04"
# number = "04"
# folder_path = f"dbfs:/student-groups/Group_{section}_{number}"
# dbutils.fs.mkdirs(folder_path)

folder_path = f"dbfs:/student-groups/Group_04_04"

# Save df_southwest_cleaned as a parquet file
# df_southwest_cleaned.write.parquet(f"{folder_path}/df_otpw_12M_southwest_cleaned_updated.parquet") # use to save a new version
df_southwest_cleaned.write.mode('overwrite').parquet(f"{folder_path}/df_otpw_12M_southwest_cleaned_updated_Apr_02.parquet") # use if you want to overwrite exisiting file

# COMMAND ----------

data_BASE_DIR = "dbfs:/student-groups/Group_04_04/"
display(dbutils.fs.ls(f"{data_BASE_DIR}"))

# COMMAND ----------

df_southwest_cleaned = spark.read.parquet(f"dbfs:/student-groups/Group_04_04/df_otpw_12M_southwest_cleaned_updated_Apr_02.parquet/")

# COMMAND ----------

# MAGIC %md
# MAGIC # Basic Graphs - we try to find patterns to report back to Southwest
# MAGIC
# MAGIC Below is a high level of what the plots are trying to achieve - 
# MAGIC After getting data only for southwest, we start with the following analysis
# MAGIC
# MAGIC 1. We show first among the total flights for the timeperiod how many were ontime, delayed and cancelled. from this we will put forth the proposal that since we have so little cancelled flights we wont be proceeding with not utilising them as part of the analysis and will keep the target as delayed by 15 mins or more.
# MAGIC
# MAGIC 2. After this we show monthly delay trends, to see if there is any seasonality, apart from that we do the same for hour of the day and day of week to find similar patterns. 
# MAGIC 3. After that Heatmap of delays by day of week and hour (using DAY_OF_WEEK, CRS_DEP_TIME, DEP_DEL15), I want to see if something emerges out of it.
# MAGIC
# MAGIC 4. After this a US map, where we calculate the percentage of delayed flight for each airport served by the airline and colour airports on an intensity scale so that it is immediately visible which airports have the max delay, also marked the airports legend based on the orign_type ( this shows whether it is a big, medium or small airport) 
# MAGIC
# MAGIC 5. Then for top 10 airports with the max delays, we see which routes are the ones suffering the most, 
# MAGIC 6. Then we go to delay factors, just a comparison of the different delay reasons and see which one causes the most delays
# MAGIC 7. Then correlation between key weather metrics vs delay probability to see which one is the most correlated of them all 
# MAGIC 8. Next we see if distance has anything to do with delays, we do a graph between distance group and delays, the index should show the value of the distance group, i.e. what distance range does each group belong to
# MAGIC

# COMMAND ----------

# 1.1 Flight Status Distribution (On-time, Delayed, Cancelled)
# Create a new column for flight status
# Create a new column for flight status
df_southwest_cleaned = df_southwest_cleaned.withColumn(
    "FLIGHT_STATUS", 
    when(col("CANCELLED") == 1, "Cancelled")
    .when(col("DEP_DEL15") == 1, "Delayed")
    .otherwise("On-time")
)

# Convert to pandas for visualization
flight_status_counts = df_southwest_cleaned.groupBy("FLIGHT_STATUS").count().toPandas()
flight_status_counts['percentage'] = flight_status_counts['count'] / flight_status_counts['count'].sum() * 100

# Plot flight status distribution (Pie Chart)
plt.figure(figsize=(8, 8))

# Define pastel colors for each status
pastel_colors = {
    'On-time': '#a8e6cf',  # pastel green
    'Delayed': '#ffb347',  # pastel orange
    'Cancelled': '#ffafaf'  # pastel red
}

# Map colors to statuses
status_colors = [pastel_colors[status] for status in flight_status_counts['FLIGHT_STATUS']]

plt.pie(
    flight_status_counts['count'], 
    labels=flight_status_counts['FLIGHT_STATUS'], 
    autopct='%1.1f%%', 
    startangle=140, 
    colors=status_colors,
    textprops={'fontsize': 12}
)
plt.title('Southwest Airlines Flight Status Distribution (12M 2015)', fontsize=15)

# Save the pie chart
plt.tight_layout()
plt.savefig('flight_status_pie_chart.png')
plt.show()

# COMMAND ----------

# 2.1 Monthly Delay Trends
monthly_delays = df_southwest_cleaned.groupBy("MONTH").agg(
    avg(col("DEP_DEL15") * 100).alias("delay_percentage"),
    count("*").alias("total_flights")
).orderBy("MONTH").toPandas()

# Convert 'MONTH' column to integers
monthly_delays['MONTH'] = monthly_delays['MONTH'].astype(int)

# Apply calendar.month_name function
monthly_delays['month_name'] = monthly_delays['MONTH'].apply(lambda x: calendar.month_name[x])

# Create a stacked bar chart
plt.figure(figsize=(12, 6))
bars = plt.bar(monthly_delays['month_name'], monthly_delays['total_flights'], color='lightgray', label='Total Flights')
plt.bar(monthly_delays['month_name'], 
        monthly_delays['delay_percentage'] / 100 * monthly_delays['total_flights'], 
        color='skyblue', label='Delayed Flights')

# Annotate delay percentages on bars
for i, row in monthly_delays.iterrows():
    plt.text(i, row['total_flights'] + 500, f"{row['delay_percentage']:.1f}%", ha='center', fontsize=10)

# Add labels and title
plt.title('Monthly Flight Delay Trends for Southwest Airlines (12M 2015)', fontsize=15)
plt.xlabel('Month', fontsize=12)
plt.ylabel('Number of Flights', fontsize=12)
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.xticks(rotation=45)

# Save the pie chart
plt.tight_layout()
plt.savefig('Monthly_Flight_Delay_Trends.png')
plt.show()

# COMMAND ----------

# 2.2 Hour of Day Delay Trends
df_southwest_cleaned = df_southwest_cleaned.withColumn("HOUR_OF_DAY", (col("CRS_DEP_TIME") / 100).cast("int"))

hourly_delays = df_southwest_cleaned.groupBy("HOUR_OF_DAY").agg(
    avg(col("DEP_DEL15") * 100).alias("delay_percentage"),
    count("*").alias("total_flights")
).orderBy("HOUR_OF_DAY").toPandas()

plt.figure(figsize=(12, 6))

# Stacked bar chart for hourly trends
bars = plt.bar(hourly_delays['HOUR_OF_DAY'], hourly_delays['total_flights'], color='lightgray', label='Total Flights')
plt.bar(hourly_delays['HOUR_OF_DAY'], 
        hourly_delays['delay_percentage'] / 100 * hourly_delays['total_flights'], 
        color='skyblue', label='Delayed Flights')

# Annotate delay percentages on bars
for i, row in hourly_delays.iterrows():
    plt.text(row['HOUR_OF_DAY'], row['total_flights'] + 100, f"{row['delay_percentage']:.1f}%", ha='center', fontsize=10)

# Add labels and title
plt.title('Hourly Flight Delay Trends for Southwest Airlines (12M 2015)', fontsize=15)
plt.xlabel('Hour of Day (24-hour format)', fontsize=12)
plt.ylabel('Number of Flights', fontsize=12)
plt.legend(loc='upper right')
plt.xticks(range(0, 24))

# Save the pie chart
plt.tight_layout()
plt.savefig('Hourly_Flight_Delay_Trends.png')
plt.show()

# COMMAND ----------

# 2.3 Day of Week Delay Trends
day_of_week_delays = df_southwest_cleaned.groupBy("DAY_OF_WEEK").agg(
    avg(col("DEP_DEL15") * 100).alias("delay_percentage"),
    count("*").alias("total_flights")
).orderBy("DAY_OF_WEEK").toPandas()

# Convert DAY_OF_WEEK to integers
day_of_week_delays['DAY_OF_WEEK'] = day_of_week_delays['DAY_OF_WEEK'].astype(int)

day_of_week_delays['day_name'] = day_of_week_delays['DAY_OF_WEEK'].apply(lambda x: calendar.day_name[x-1])

# Create a stacked bar chart
plt.figure(figsize=(12, 6))
bars = plt.bar(day_of_week_delays['day_name'], day_of_week_delays['total_flights'], color='lightgray', label='Total Flights')
plt.bar(day_of_week_delays['day_name'], 
        day_of_week_delays['delay_percentage'] / 100 * day_of_week_delays['total_flights'], 
        color='skyblue', label='Delayed Flights')

# Annotate delay percentages on bars
for i, row in day_of_week_delays.iterrows():
    plt.text(i, row['total_flights'] + 500, f"{row['delay_percentage']:.1f}%", ha='center', fontsize=10)

# Add labels and title
plt.title('Day of Week Flight Delay Trends for Southwest Airlines (12M 2015)', fontsize=15)
plt.xlabel('Day of Week', fontsize=12)
plt.ylabel('Number of Flights', fontsize=12)
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

# Save the pie chart
plt.tight_layout()
plt.savefig('Day_of_week_Flight_Delay_Trends.png')
plt.show()

# COMMAND ----------

# 3. Heatmap of delays by day of week and hour
day_hour_delays = df_southwest_cleaned.groupBy("DAY_OF_WEEK", "HOUR_OF_DAY").agg(
    avg(col("DEP_DEL15") * 100).alias("delay_percentage")
).toPandas()

# Create a pivot table for the heatmap
heatmap_data = day_hour_delays.pivot(index='DAY_OF_WEEK', columns='HOUR_OF_DAY', values='delay_percentage')

# Map day numbers to day names
day_names = {1: 'Monday', 2: 'Tuesday', 3: 'Wednesday', 4: 'Thursday', 5: 'Friday', 6: 'Saturday', 7: 'Sunday'}
heatmap_data.index = [day_names[int(day)] for day in heatmap_data.index]

plt.figure(figsize=(14, 8))
sns.heatmap(heatmap_data, cmap='YlOrRd', annot=True, fmt='.1f', linewidths=.5)
plt.title('Delay Percentage by Day of Week and Hour of Day for Southwest Airlines (12M 2015)', fontsize=15)
plt.xlabel('Hour of Day (24-hour format)', fontsize=12)
plt.ylabel('Day of Week', fontsize=12)

# Save the pie chart
plt.tight_layout()
plt.savefig('Heatmap_Delay_Trends.png')
plt.show()

# COMMAND ----------

# Calculate delay percentage for each origin airport
delay_analysis = df_southwest_cleaned.groupby(
    'ORIGIN', 'origin_airport_lat', 'origin_airport_lon', 'origin_type'
).agg(
    (F.mean('DEP_DEL15') * 100).alias('delay_percentage'),
    F.count('DEP_DEL15').alias('total_flights')
).toPandas()

# Create a map centered on the US
us_map = folium.Map(location=[39.8283, -98.5795], zoom_start=4)

# Create a color scale
color_scale = LinearColormap(['#FFCCCC', '#FF0000'], vmin=delay_analysis['delay_percentage'].min(), vmax=delay_analysis['delay_percentage'].max())

# Add the color scale to the map
color_scale.add_to(us_map)

# Add circle markers for each airport (no pins, just circles with color intensity)
for _, row in delay_analysis.iterrows():
    # Determine size based on airport type
    if row['origin_type'] == 'large_airport':
        radius = 8
    elif row['origin_type'] == 'medium_airport':
        radius = 6
    else:  # small_airport
        radius = 4
        
    # Create popup text
    popup_text = f"""
    <b>Airport:</b> {row['ORIGIN']}<br>
    <b>Type:</b> {row['origin_type']}<br>
    <b>Delay %:</b> {row['delay_percentage']:.1f}%<br>
    <b>Total Flights:</b> {row['total_flights']}
    """
    
    # Add circle marker (no pin icon)
    folium.CircleMarker(
        location=[row['origin_airport_lat'], row['origin_airport_lon']],
        radius=radius,
        color='black',
        weight=1,
        fill=True,
        fill_color=color_scale(row['delay_percentage']),
        fill_opacity=0.7,
        popup=folium.Popup(popup_text, max_width=300)
    ).add_to(us_map)

# Add a legend for the top 10 airports with the most delays
top_10_airports = delay_analysis.sort_values(by='delay_percentage', ascending=False).head(10)

legend_html = '''
<div style="position: fixed; bottom: 50px; right: 50px; width: 300px; height: auto; border:2px solid grey; z-index:9999; font-size:14px; background-color:white; padding: 10px; border-radius: 5px;">
<b>Top 10 Airports with Most Delays (12M 2015)</b><br>
'''
for i, row in enumerate(top_10_airports.itertuples(), 1):
    # Add airport type indicator
    airport_type_indicator = "L" if row.origin_type == 'large_airport' else "M" if row.origin_type == 'medium_airport' else "S"
    legend_html += f"{i}. {row.ORIGIN} ({airport_type_indicator}) - {row.delay_percentage:.1f}%<br>"
legend_html += '''
<br><span style="font-size:12px;">Airport Type: L = Large, M = Medium, S = Small</span>
</div>
'''

us_map.get_root().html.add_child(folium.Element(legend_html))

# Save the map
us_map.save('southwest_airport_delays_updated.html')

# COMMAND ----------

# Display the map
display(us_map)

# COMMAND ----------

# Create a figure with 10 subplots (two rows of 5 for each airport)
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
axes = axes.flatten()  # Flatten the 2D array to make indexing easier
fig.suptitle('Top 10 Delay-Prone Routes for Highest-Delay Airports (12M 2015)', fontsize=16)

# For each top airport, find its most delayed routes
for i, (_, airport) in enumerate(top_10_airports.iterrows()):
    # Filter for flights from this origin
    airport_routes = df_southwest_cleaned.filter(col("ORIGIN") == airport['ORIGIN'])
    
    # Group by destination and calculate delay percentage
    route_delays = airport_routes.groupBy("DEST").agg(
        (avg(col("DEP_DEL15")) * 100).alias("delay_percentage"),
        count("*").alias("total_flights")
    ).filter(col("total_flights") > 10).orderBy(desc("delay_percentage")).limit(5).toPandas()
    
    # Create small map for this airport's routes
    ax = axes[i]
    ax.set_title(f"{airport['ORIGIN']} ({airport['origin_type'][0].upper()})")
    
    # Plot bar chart of top 5 routes
    bars = ax.bar(route_delays['DEST'], route_delays['delay_percentage'])
    ax.set_ylabel('Delay %')
    ax.set_xlabel('Destination')
    ax.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.subplots_adjust(top=0.90)
plt.show()

# Step 5: Create a map visualization showing the top 10 airports and their routes
us_map = folium.Map(location=[39.8283, -98.5795], zoom_start=4)

# Create a colormap for the routes
colors = cm.rainbow(np.linspace(0, 1, len(top_10_airports)))
airport_colors = {}

# Convert RGB to hex colors for Folium
for i, (_, airport) in enumerate(top_10_airports.iterrows()):
    hex_color = '#{:02x}{:02x}{:02x}'.format(
        int(colors[i][0]*255), 
        int(colors[i][1]*255), 
        int(colors[i][2]*255)
    )
    airport_colors[airport['ORIGIN']] = hex_color

# Add markers for the top 10 airports
for i, (_, airport) in enumerate(top_10_airports.iterrows()):
    # Get color for this airport
    airport_color = airport_colors[airport['ORIGIN']]
    
    # Add marker for the airport
    folium.CircleMarker(
        location=[float(airport['origin_airport_lat']), float(airport['origin_airport_lon'])],
        radius=10,
        color=airport_color,
        fill=True,
        fill_color=airport_color,
        fill_opacity=0.7,
        popup=f"<b>{airport['ORIGIN']}</b><br>Delay %: {airport['delay_percentage']:.1f}%<br>Type: {airport['origin_type']}"
    ).add_to(us_map)
    
    # Get top 5 delayed routes for this airport
    airport_routes = df_southwest_cleaned.filter(col("ORIGIN") == airport['ORIGIN'])
    top_routes = airport_routes.groupBy("DEST", "dest_airport_lat", "dest_airport_lon").agg(
        (avg(col("DEP_DEL15")) * 100).alias("delay_percentage"),
        count("*").alias("total_flights")
    ).filter(col("total_flights") > 10).orderBy(desc("delay_percentage")).limit(5).toPandas()
    
    # Draw lines for each route with the airport's color
    for _, route in top_routes.iterrows():
        folium.PolyLine(
            locations=[
                [float(airport['origin_airport_lat']), float(airport['origin_airport_lon'])],
                [float(route['dest_airport_lat']), float(route['dest_airport_lon'])]
            ],
            color=airport_color,
            weight=2,
            opacity=0.6,
            popup=f"{airport['ORIGIN']} to {route['DEST']}: {route['delay_percentage']:.1f}% delays"
        ).add_to(us_map)
        
        # Add a small marker for the destination
        folium.CircleMarker(
            location=[float(route['dest_airport_lat']), float(route['dest_airport_lon'])],
            radius=4,
            color='green',
            fill=True,
            fill_color='green',
            fill_opacity=0.7,
            popup=f"<b>{route['DEST']}</b>"
        ).add_to(us_map)

# Add a legend for the top 10 airports
legend_html = '''
<div style="position: fixed; bottom: 50px; right: 50px; width: 250px; height: auto; 
    border:2px solid grey; z-index:9999; font-size:14px; background-color:white; 
    padding: 10px; border-radius: 5px;">
<b>Top 10 Airports with Highest Delays (12M 2015)</b><br>
'''

for i, (_, airport) in enumerate(top_10_airports.iterrows()):
    airport_type = "L" if airport['origin_type'] == "large_airport" else "M" if airport['origin_type'] == "medium_airport" else "S"
    airport_color = airport_colors[airport['ORIGIN']]
    legend_html += f'<span style="color:{airport_color}">●</span> {i+1}. {airport["ORIGIN"]} ({airport_type}) - {airport["delay_percentage"]:.1f}%<br>'

legend_html += '''
<br><span style="font-size:12px;">Airport Type: L = Large, M = Medium, S = Small</span>
</div>
'''

us_map.get_root().html.add_child(folium.Element(legend_html))

# Save the map
us_map.save('top_10_delayed_airports_routes.html')

# COMMAND ----------

display(us_map)

# COMMAND ----------

# 6. Delay Factors Comparison by Month
# Filter for delayed flights only
delayed_flights = df_southwest_cleaned.filter(col("DEP_DEL15") == 1)

# Calculate percentage contribution of each delay reason to total delay minutes
delay_reasons_monthly = delayed_flights.groupBy("MONTH").agg(
    (spark_sum(col("CARRIER_DELAY")) / 
     spark_sum(col("CARRIER_DELAY") + col("WEATHER_DELAY") + col("NAS_DELAY") + 
         col("SECURITY_DELAY") + col("LATE_AIRCRAFT_DELAY")) * 100).alias("Carrier Delay"),
    (spark_sum(col("WEATHER_DELAY")) / 
     spark_sum(col("CARRIER_DELAY") + col("WEATHER_DELAY") + col("NAS_DELAY") + 
         col("SECURITY_DELAY") + col("LATE_AIRCRAFT_DELAY")) * 100).alias("Weather Delay"),
    (spark_sum(col("NAS_DELAY")) / 
     spark_sum(col("CARRIER_DELAY") + col("WEATHER_DELAY") + col("NAS_DELAY") + 
         col("SECURITY_DELAY") + col("LATE_AIRCRAFT_DELAY")) * 100).alias("National Air System Delay"),
    (spark_sum(col("SECURITY_DELAY")) / 
     spark_sum(col("CARRIER_DELAY") + col("WEATHER_DELAY") + col("NAS_DELAY") + 
         col("SECURITY_DELAY") + col("LATE_AIRCRAFT_DELAY")) * 100).alias("Security Delay"),
    (spark_sum(col("LATE_AIRCRAFT_DELAY")) / 
     spark_sum(col("CARRIER_DELAY") + col("WEATHER_DELAY") + col("NAS_DELAY") + 
         col("SECURITY_DELAY") + col("LATE_AIRCRAFT_DELAY")) * 100).alias("Late Aircraft Delay"),
    count("*").alias("total_delayed_flights")
).orderBy("MONTH").toPandas()

# Ensure 'MONTH' column is of integer type
delay_reasons_monthly['MONTH'] = delay_reasons_monthly['MONTH'].astype(int)

delay_reasons_monthly['month_name'] = delay_reasons_monthly['MONTH'].apply(lambda x: calendar.month_name[x])

# Create a stacked bar chart with lighter colors
plt.figure(figsize=(14, 8))

# Use a lighter color palette
colors = plt.cm.Pastel1(np.linspace(0, 0.8, 5))  # Pastel colors

delay_reasons_monthly.set_index('month_name')[['Carrier Delay', 'Weather Delay', 'National Air System Delay', 
                                              'Security Delay', 'Late Aircraft Delay']].plot(
    kind='bar', 
    stacked=True,
    color=colors
)

# Add labels and title
plt.title('Percentage Contribution of Delay Reasons by Month for Southwest Airlines (12M 2015)', fontsize=15)
plt.xlabel('Month', fontsize=12)
plt.ylabel('Percentage of Total Delay Minutes', fontsize=12)
plt.legend(title='Delay Reason', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Set y-axis to show 0-100%
plt.ylim(0, 100)

# Save the pie chart
plt.tight_layout()
plt.savefig('Delay_Reason.png')
plt.show()

# COMMAND ----------

# 7.1 Correlation between Weather Metrics and Delay Probability
weather_columns = [
 'HourlyDewPointTemperature',
 'HourlyDryBulbTemperature',
 'HourlyPrecipitation',
 'HourlyRelativeHumidity',
 'HourlySkyConditions',
 'HourlySeaLevelPressure',
 'HourlyVisibility',
 'HourlyWindDirection',
 'HourlyWindSpeed',
]

# Convert columns to numeric if they are not already
for col_name in weather_columns + ["DEP_DEL15"]:
    df_southwest_cleaned = df_southwest_cleaned.withColumn(col_name, col(col_name).cast("double"))

# Calculate correlation between each weather metric and delay probability
weather_correlations = []
for col_name in weather_columns:
    correlation = df_southwest_cleaned.stat.corr(col_name, "DEP_DEL15")
    weather_correlations.append({"Weather Metric": col_name, "Correlation with Delay": correlation})

# Convert to Pandas DataFrame for plotting
weather_corr_df = pd.DataFrame(weather_correlations)

# Plotting
plt.figure(figsize=(12, 6))
sns.barplot(x='Weather Metric', y='Correlation with Delay', data=weather_corr_df, palette='coolwarm')
plt.title('Correlation between Weather Metrics and Flight Delay Probability (12M 2015)', fontsize=15)
plt.xlabel('Weather Metric', fontsize=12)
plt.ylabel('Correlation Coefficient', fontsize=12)
plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
plt.xticks(rotation=45)

# Save the pie chart
plt.tight_layout()
plt.savefig('Correlation_between_Weather_Metrics.png')
plt.show()

# COMMAND ----------

# 8. Distance Group vs Delay Probability
# Calculate the percentage of delayed flights for each distance group
distance_group_analysis = df_southwest_cleaned.groupby('DISTANCE_GROUP').agg(
    (F.mean('DEP_DEL15') * 100).alias('delayed_percentage'),
    F.count('DEP_DEL15').alias('total_flights'),
    F.min('DISTANCE').alias('min_distance'),
    F.max('DISTANCE').alias('max_distance')
).toPandas()

# Add a column for the distance range
distance_group_analysis['distance_range'] = distance_group_analysis.apply(
    lambda row: f"{int(float(row['min_distance']))}-{int(float(row['max_distance']))} miles" if row['max_distance'] != row['min_distance'] else f"{int(float(row['min_distance']))} miles",
    axis=1
)

# Create a stacked bar chart
plt.figure(figsize=(12, 6))

# Bar chart for total flights
bars = plt.bar(distance_group_analysis['distance_range'], 
               distance_group_analysis['total_flights'], 
               color='lightgray', label='Total Flights')

# Bar chart for delayed flights
plt.bar(distance_group_analysis['distance_range'], 
        distance_group_analysis['delayed_percentage'] / 100 * distance_group_analysis['total_flights'], 
        color='skyblue', label='Delayed Flights')

# Annotate delay percentages on bars
for i, row in distance_group_analysis.iterrows():
    plt.text(i, row['total_flights'] + 50, f"{row['delayed_percentage']:.1f}%", ha='center', fontsize=10)

# Add labels and title
plt.title('Flight Delay Trends by Distance Group for Southwest Airlines (12M 2015)', fontsize=15)
plt.xlabel('Distance Range', fontsize=12)
plt.ylabel('Number of Flights', fontsize=12)
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.xticks(rotation=45, ha='right')

# Save the pie chart
plt.tight_layout()
plt.savefig('Delay_vs_Distance.png')
plt.show()

# COMMAND ----------

# Load the filtered dataset (with DEP_DEL15 != 2)
df_filtered = df_southwest_cleaned

# Convert DecimalType columns to float in Spark before conversion
weather_cols = [
    "HourlyDewPointTemperature", "HourlyDryBulbTemperature", "HourlyPrecipitation",
    "HourlyRelativeHumidity", "HourlySeaLevelPressure", "HourlyVisibility",
    "HourlyWindDirection", "HourlyWindSpeed"
]

for col in weather_cols:
    if col in df_filtered.columns:
        df_filtered = df_filtered.withColumn(col, F.col(col).cast("float"))

# Convert the Spark DataFrame to a pandas-on-Spark DataFrame
df_filtered_ps = ps.DataFrame(df_filtered)

# Select numerical features for correlation analysis
numerical_features = [
    "DEP_DEL15", "YEAR", "QUARTER", "MONTH", 
    "DAY_OF_WEEK", "DAY_OF_MONTH", "CRS_DEP_TIME", "CRS_ARR_TIME",
    "DISTANCE", "DISTANCE_GROUP", "origin_airport_lat", "origin_airport_lon", 
    "dest_airport_lat", "dest_airport_lon", "HourlyDewPointTemperature", 
    "HourlyDryBulbTemperature", "HourlyPrecipitation", "HourlyVisibility", 
    "HourlySeaLevelPressure", "HourlyWindDirection", "HourlyWindSpeed"
]

# Filter to include only columns that exist in the dataset
numerical_features = [col for col in numerical_features if col in df_filtered_ps.columns]

# Convert to pandas DataFrame
df_filtered_pd = df_filtered_ps.to_pandas()

# Calculate the correlation matrix
correlation_matrix = df_filtered_pd[numerical_features].corr(numeric_only=True)

# Create correlation heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Pearson Correlation Heatmap of Selected Features (12M 2015)', fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
plt.show()

# COMMAND ----------

# Create scatter plots for each feature vs DEP_DEL15
features_to_plot = [f for f in numerical_features if f != "DEP_DEL15"]
num_features = len(features_to_plot)
num_cols = 4
num_rows = (num_features + num_cols - 1) // num_cols

plt.figure(figsize=(20, num_rows * 5))

for i, feature in enumerate(features_to_plot):
    plt.subplot(num_rows, num_cols, i+1)
    
    # Add some jitter to the binary target for better visualization
    y = df_filtered_pd['DEP_DEL15'] + np.random.normal(0, 0.02, size=len(df_filtered_pd))
    
    # Create scatter plot
    plt.scatter(df_filtered_pd[feature], y, alpha=0.3, s=10)
    
    # Add a trend line to help visualize patterns
    try:
        # Calculate mean DEP_DEL15 for binned feature values
        bins = pd.cut(df_filtered_pd[feature], 20)
        mean_by_bin = df_filtered_pd.groupby(bins)['DEP_DEL15'].mean()
        bin_centers = [(x.left + x.right)/2 for x in mean_by_bin.index]
        plt.plot(bin_centers, mean_by_bin.values, 'r-', linewidth=2)
    except:
        pass  # Skip trend line if binning fails
    
    plt.title(f'{feature} vs DEP_DEL15 (12M 2015)', fontsize=12)
    plt.xlabel(feature)
    plt.ylabel('DEP_DEL15')
    plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('scatter_plots_target.png', dpi=300)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # Read the Processed Data and Start Data Engineering and Feature Selection

# COMMAND ----------

df_southwest_cleaned_training = spark.read.parquet(f"dbfs:/student-groups/Group_04_04/df_otpw_12M_southwest_cleaned_updated_Apr_02.parquet/")

# COMMAND ----------

null_percentage_df = calculate_null_percentage(df_southwest_cleaned_training)
display(null_percentage_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Final features for modelling - picked based on EDA

# COMMAND ----------

selected_features_for_training = [
    "DEP_DEL15", "FL_DATE", "YEAR", "QUARTER", "MONTH", 
    "DAY_OF_WEEK", "DAY_OF_MONTH", "ORIGIN", "DEST", "CRS_DEP_TIME", "origin_type",
    "dest_type", "origin_airport_lat", "origin_airport_lon", "dest_airport_lat", "dest_airport_lon", "DISTANCE_GROUP",
    "HourlyDewPointTemperature", "HourlyDryBulbTemperature", "HourlyPrecipitation", "HourlyRelativeHumidity", "HourlySeaLevelPressure",
    "HourlyVisibility", "HourlyWindDirection", "HourlyWindSpeed"
]

# COMMAND ----------

# MAGIC %md
# MAGIC #### Data Engineering - adding additional features

# COMMAND ----------

# First, select only the features we need for training
df_southwest_cleaned_training = df_southwest_cleaned_training.select(selected_features_for_training)

# Filter out cancelled flights (DEP_DEL15 = 2)
df_southwest_cleaned_training = df_southwest_cleaned_training.filter(df_southwest_cleaned_training["DEP_DEL15"] != 2)

# Create a Year-Quarter column
df_southwest_cleaned_training = df_southwest_cleaned_training.withColumn("YEAR_QUARTER", 
    F.concat(F.col("YEAR"), F.lit("-Q"), F.col("QUARTER")))

# Extract hour from CRS_DEP_TIME for better feature representation
df_southwest_cleaned_training = df_southwest_cleaned_training.withColumn("DEP_HOUR", 
    (F.col("CRS_DEP_TIME") / 100).cast(IntegerType()))

# Extract minute from CRS_DEP_TIME for better feature representation
df_southwest_cleaned_training = df_southwest_cleaned_training.withColumn(
    "DEP_MINUTE", 
    F.col("CRS_DEP_TIME") % 100
)
# Create time buckets (15-minute intervals) for minutes, instead of taking everything
df_southwest_cleaned_training = df_southwest_cleaned_training.withColumn(
    "DEP_TIME_BUCKET", 
    F.concat(
        F.col("DEP_HOUR").cast("string"), 
        F.lit(":"), 
        F.floor(F.col("DEP_MINUTE") / 15) * 15
    )
)

# Add a weekend variable
df_southwest_cleaned_training = df_southwest_cleaned_training.withColumn("is_weekend", when(df_southwest_cleaned_training.DAY_OF_WEEK.isin(6, 7), 1).otherwise(0))

# Route-specific features
df_southwest_cleaned_training = df_southwest_cleaned_training.withColumn(
    "ROUTE", 
    F.concat(F.col("ORIGIN"), F.lit("-"), F.col("DEST"))
)

# Time of day categories
df_southwest_cleaned_training = df_southwest_cleaned_training.withColumn(
    "TIME_OF_DAY", 
    F.when(F.col("DEP_HOUR").between(5, 9), "morning")
    .when(F.col("DEP_HOUR").between(10, 15), "midday")
    .when(F.col("DEP_HOUR").between(16, 19), "evening")
    .otherwise("night")
)

# Day-Hour interaction
df_southwest_cleaned_training = df_southwest_cleaned_training.withColumn(
    "DAY_HOUR", 
    F.concat(F.col("DAY_OF_WEEK"), F.lit("_"), F.col("DEP_HOUR"))
)

# Holiday season indicator (derived from the EDA that we got)
df_southwest_cleaned_training = df_southwest_cleaned_training.withColumn(
    "holiday_season", 
    F.when(F.col("MONTH").isin(6, 7, 12), 1).otherwise(0)
)

# Calculate the number of flights per airport per day
window_airport_date = Window.partitionBy("ORIGIN", "FL_DATE")
df_southwest_cleaned_training = df_southwest_cleaned_training.withColumn(
    "flights_per_origin_day", 
    F.count("*").over(window_airport_date)
)


# Creates a rolling 7-day delay history for each route by
# window_route = Window.partitionBy("ROUTE").orderBy("FL_DATE").rowsBetween(-7, -1)
# df_southwest_cleaned_training = df_southwest_cleaned_training.withColumn(
#     "route_7day_delay_rate", 
#     F.avg("DEP_DEL15").over(window_route)
# )

# Calculate origin airport average delay rate
airport_delay_rates = df_southwest_cleaned_training.groupBy("ORIGIN").agg(
    F.avg("DEP_DEL15").alias("origin_delay_rate")
)

# Join and fill nulls with origin airport delay rate ( this is because the 7 day rolling col has nulls)
df_southwest_cleaned_training = df_southwest_cleaned_training.join(
    airport_delay_rates, on="ORIGIN", how="left"
)

# df_southwest_cleaned_training = df_southwest_cleaned_training.withColumn(
#     "route_7day_delay_rate",
#     F.coalesce(F.col("route_7day_delay_rate"), F.col("origin_delay_rate"))
# )

# Each airport's overall traffic volume, total flights leaving from each airport around the year (since this is 12 M data)
airports_total_flights = df_southwest_cleaned_training.groupBy("ORIGIN").count().withColumnRenamed("count", "total_flights")
df_southwest_cleaned_training = df_southwest_cleaned_training.join(
    airports_total_flights, 
    "ORIGIN", 
    "left"
)
df_southwest_cleaned_training = df_southwest_cleaned_training.withColumn(
    "airport_congestion_level", 
    F.col("total_flights") / F.lit(1000)
)

# Create categorical buckets for airports based on delay rates (used for stratification)
df_southwest_cleaned_training = df_southwest_cleaned_training.withColumn(
    "airport_delay_category",
    F.when(F.col("origin_delay_rate") < 0.15, "low_delay")
    .when(F.col("origin_delay_rate") < 0.25, "medium_delay")
    .otherwise("high_delay")
)

# Analyze missing values in cleaned features
null_percentage_df = calculate_null_percentage(df_southwest_cleaned_training)
display(null_percentage_df)

# COMMAND ----------

display(df_southwest_cleaned_training)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Pearson Correlation Matrix for Numerical and Temporal Features

# COMMAND ----------

# Combine numerical and temporal features
numerical_features_with_temporal = [
    "origin_airport_lat", "origin_airport_lon", 
    "dest_airport_lat", "dest_airport_lon",
    # Weather and operational metrics
    "HourlyDewPointTemperature", "HourlyDryBulbTemperature",
    "HourlyPrecipitation", "HourlyRelativeHumidity",
    "HourlySeaLevelPressure", "HourlyVisibility",
    "HourlyWindDirection", "HourlyWindSpeed",
    # Engineered features
    "is_weekend", "holiday_season",
    "flights_per_origin_day",
    # "route_7day_delay_rate",
    "origin_delay_rate",
    "total_flights",
    "airport_congestion_level",
    # Add temporal features
    "QUARTER", "MONTH", 
    "DAY_OF_WEEK", "DAY_OF_MONTH",
     "DEP_HOUR", "DEP_MINUTE",
]

# Create vector column for correlation calculation
assembler = VectorAssembler(inputCols=numerical_features_with_temporal, outputCol="features_vector")
df_vector = assembler.transform(df_southwest_cleaned_training).select("features_vector")

# Compute correlation matrix
correlation_matrix = Correlation.corr(df_vector, "features_vector").head()[0]
corr_array = correlation_matrix.toArray()

# Convert to Pandas DataFrame for visualization
corr_df = pd.DataFrame(corr_array, index=numerical_features_with_temporal, columns=numerical_features_with_temporal)

# Plot heatmap
plt.figure(figsize=(14, 12))
sns.heatmap(corr_df, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Pearson Correlation Matrix for Numerical and Temporal Features")

plt.tight_layout()
plt.show()

# COMMAND ----------

# Split the dataset
# Test set: Last quarter (Q4)
test_data = df_southwest_cleaned_training.filter(F.col("QUARTER") == 4)

# Validation set: Two months before the test set (October and November)
validation_data = df_southwest_cleaned_training.filter((F.col("MONTH") == 8) | (F.col("MONTH") == 9))

# Train set: All other data
train_data = df_southwest_cleaned_training.filter((F.col("QUARTER") != 4) & (F.col("MONTH") != 8) & (F.col("MONTH") != 9))

print(f"Train data count: {train_data.count()}")
print(f"Validation data count: {validation_data.count()}")
print(f"Test data count: {test_data.count()}")

# COMMAND ----------

# Visualize delayed vs on-time flights for each dataset
plt.figure(figsize=(15, 5))

# Function to plot delayed vs on-time flights
def plot_delay_distribution(data, position, title):
    delay_counts = data.groupBy("DEP_DEL15").count().toPandas()
    delay_counts = delay_counts.sort_values("DEP_DEL15")
    
    # Calculate percentages
    total = delay_counts["count"].sum()
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
plot_delay_distribution(train_data, 1, "Train Data")
plot_delay_distribution(validation_data, 2, "Validation Data")
plot_delay_distribution(test_data, 3, "Test Data")

plt.tight_layout()
plt.show()

# COMMAND ----------

# Calculate sample weights
total_count = train_data.count()
delayed_count = train_data.filter(col("DEP_DEL15") == 1).count()
on_time_count = total_count - delayed_count  # For class 0

delayed_weight = total_count / (2 * delayed_count)
on_time_weight = total_count / (2 * on_time_count)

# Add weight column to training data
train_data = train_data.withColumn(
    "sample_weight",
    when(col("DEP_DEL15") == 1, delayed_weight).otherwise(on_time_weight)
)

# COMMAND ----------

folder_path = f"dbfs:/student-groups/Group_04_04"

# Save train/validate/test split as a parquet files
train_data.write.parquet(f"{folder_path}/df_otpw_12M_train_data.parquet") # use to save a new version
validation_data.write.parquet(f"{folder_path}/df_otpw_12M_validation_data.parquet") # use to save a new version
test_data.write.parquet(f"{folder_path}/df_otpw_12M_test_data.parquet") # use to save a new version
# df_southwest_cleaned.write.mode('overwrite').parquet(f"{folder_path}/df_otpw_12M_southwest_cleaned_updated_Apr_02.parquet") # use if you want to overwrite exisiting file

# COMMAND ----------

# MAGIC %md
# MAGIC ### MLLib pipeline starts

# COMMAND ----------

# MAGIC %md
# MAGIC Please note that for conveniece, we are evaluating test datasets for each model such that if one model is chosen, we don't need to run the costly model training and evaluation to see its test model, better efficiency and coordination. However, similar as what we did in cross validation, when evaluating the model, we only consider validation evaluation metrics.

# COMMAND ----------

# Will use in phase 3
# train_data = spark.read.parquet(f"dbfs:/student-groups/Group_04_04/df_otpw_12M_train_data.parquet/")
# validation_data = spark.read.parquet(f"dbfs:/student-groups/Group_04_04/df_otpw_12M_validation_data.parquet/")
# test_data = spark.read.parquet(f"dbfs:/student-groups/Group_04_04/df_otpw_12M_test_data.parquet/")

# COMMAND ----------

# Define categorical features with few unique values (for one-hot encoding)
categorical_features = [
    "ORIGIN", "DEST", "origin_type", "dest_type",
    "YEAR_QUARTER", "DEP_TIME_BUCKET", "TIME_OF_DAY", "ROUTE",
    "DISTANCE_GROUP", "MONTH",
    "DAY_OF_WEEK", 'DAY_HOUR'
    # "DEP_HOUR", 
]

# Define numerical features for standardization
numerical_features = [
    # Weather and operational metrics
    "HourlyDewPointTemperature", "HourlyDryBulbTemperature",
    "HourlyPrecipitation", "HourlyRelativeHumidity",
    "HourlySeaLevelPressure", "HourlyVisibility",
    "HourlyWindDirection", "HourlyWindSpeed",
    # Engineered features
    "is_weekend", "holiday_season",
    "flights_per_origin_day",
    # "route_7day_delay_rate",
    "origin_delay_rate",
    # "airport_congestion_level"
]

# Create stages for the pipeline
stages = []

# Process categorical features with one-hot encoding
for feature in categorical_features:
    # Create string indexer
    indexer = StringIndexer(inputCol=feature, outputCol=f"{feature}_indexed", handleInvalid="keep")
    # Create one-hot encoder
    encoder = OneHotEncoder(inputCol=f"{feature}_indexed", outputCol=f"{feature}_encoded")
    # Add stages
    stages += [indexer, encoder]

# Collect all transformed features
transformed_features = [f"{feature}_encoded" for feature in categorical_features] + \
                       numerical_features

# COMMAND ----------

# Create vector assembler
assembler = VectorAssembler(inputCols=transformed_features, outputCol="features_unscaled", handleInvalid="keep")
stages.append(assembler)

# Create standard scaler
scaler = StandardScaler(inputCol="features_unscaled", outputCol="features", withStd=True, withMean=True)
stages.append(scaler)

# Create and fit the pipeline
pipeline = Pipeline(stages=stages)
pipeline_model = pipeline.fit(train_data)

# Transform the datasets
train_data_transformed = pipeline_model.transform(train_data)
train_data_transformed = train_data_transformed.cache()
validation_data_transformed = pipeline_model.transform(validation_data)
validation_data_transformed = validation_data_transformed.cache()
test_data_transformed = pipeline_model.transform(test_data)
test_data_transformed = test_data_transformed.cache()

# Prepare data for modeling
train_data_ml = train_data_transformed.select("DEP_DEL15", "features", "sample_weight")
validation_data_ml = validation_data_transformed.select("DEP_DEL15", "features")
test_data_ml = test_data_transformed.select("DEP_DEL15", "features")

# COMMAND ----------

# Evaluators for precision, recall, and F1 score

def evaluate_predictions(predictions, target_feature, dataset_name):
    # Calculate confusion matrix components
    tp = predictions.filter((col(target_feature) == 1) & (col('prediction') == 1)).count()
    tn = predictions.filter((col(target_feature) == 0) & (col('prediction') == 0)).count()
    fp = predictions.filter((col(target_feature) == 0) & (col('prediction') == 1)).count()
    fn = predictions.filter((col(target_feature) == 1) & (col('prediction') == 0)).count()

    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    return {
        'dataset': dataset_name,
        'precision': precision,
        'recall': recall,
        'f1': f1,
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
    print(metrics_df[['precision', 'recall', 'f1']].to_markdown(floatfmt=".3f"))

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

# Baseline: with sample weights when training, no regularization

# Logistic Regression Model for classification
lr = LogisticRegression(featuresCol="features", labelCol="DEP_DEL15", weightCol="sample_weight", maxIter=10, regParam=0)
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

# Pure Lasso Regularization with all coeffs=0, used as a sanity check. Its eval metrics are good to be compared against given we can see how much our model improves in contrast of a simple classifier predicting everything as positive.

# Logistic Regression Model for classification
lr = LogisticRegression(featuresCol="features", labelCol="DEP_DEL15", weightCol="sample_weight", maxIter=10, regParam=1, elasticNetParam=1)
lr_model = lr.fit(train_data_ml)

# Get predictions for all datasets
train_pred = lr_model.transform(train_data_ml)
val_pred = lr_model.transform(validation_data_ml)
test_pred = lr_model.transform(test_data_ml)
model_datasets = {'Training': train_pred, 'Validation': val_pred, 'Test': test_pred}
show_predictions_eval(model_datasets, "DEP_DEL15")

# Access the coefficients (weights)
coefficients = lr_model.coefficients

for coeff in coefficients:
    print(coeff)

# COMMAND ----------

# Elastic Net Regularization with a equal mix of Lasso and Ridge

# Logistic Regression Model for classification
lr = LogisticRegression(featuresCol="features", labelCol="DEP_DEL15", weightCol="sample_weight", maxIter=10, regParam=0.01, elasticNetParam=0.5)
lr_model = lr.fit(train_data_ml)

# Get predictions for all datasets
train_pred = lr_model.transform(train_data_ml)
val_pred = lr_model.transform(validation_data_ml)
test_pred = lr_model.transform(test_data_ml)
model_datasets = {'Training': train_pred, 'Validation': val_pred, 'Test': test_pred}
show_predictions_eval(model_datasets, "DEP_DEL15")

# Access the coefficients (weights)
coefficients = lr_model.coefficients

for coeff in coefficients:
    print(coeff)

# COMMAND ----------

# Pure Lasso Regularization

# Logistic Regression Model for classification
lr = LogisticRegression(featuresCol="features", labelCol="DEP_DEL15", weightCol="sample_weight", maxIter=10, regParam=0.01, elasticNetParam=1)
lr_model = lr.fit(train_data_ml)

# Get predictions for all datasets
train_pred = lr_model.transform(train_data_ml)
val_pred = lr_model.transform(validation_data_ml)
test_pred = lr_model.transform(test_data_ml)
model_datasets = {'Training': train_pred, 'Validation': val_pred, 'Test': test_pred}
show_predictions_eval(model_datasets, "DEP_DEL15")

# Access the coefficients (weights)
coefficients = lr_model.coefficients

for coeff in coefficients:
    print(coeff)

# COMMAND ----------

# Loss after each iteration
# As we can see, it stablized
lr_model.summary.objectiveHistory

# COMMAND ----------

# MAGIC %md
# MAGIC # HyperTuning and Cross Validation

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

