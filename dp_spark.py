from pyspark.sql import SparkSession
from pyspark.sql.functions import col, regexp_replace, date_format
import sys
from os import path
from general import *

@show_elapsed_time
def read_data(input_path, logger : Logger):
    # open spark session
    spark = SparkSession.builder \
            .appName("Data Preparation PySpark") \
            .config("spark.driver.memory", "4g") \
            .config("spark.executor.memory", "4g") \
            .getOrCreate()

    # read data from csv file
    df = spark.read.csv(input_path, inferSchema = True, header = True)

    # show schema of original data
    msg = "The schema of data before preparation is"
    show_schema(df, msg, logger)

    return spark, df

def show_schema(df, msg : str, logger:Logger):
    # show data schema and log it
    print(msg)
    content = capture_terminal_output(df.printSchema)
    logger.log(msg, content)

def show_query_result(result, msg : str, logger:Logger, limit : int = -1):
    # show query result and log it
    msg = " ==> ".join([msg, f"record count: {result.count()}"])
    print(msg)
    
    # show as many rows as requested
    content = capture_terminal_output(result.show, truncate=False, n=limit if limit > 0 else result.count())
    logger.log(msg, content)


@show_elapsed_time
def check_data_quality(spark, df, logger:Logger):
    # check data quality
    
    # create a view based on the dataframe
    df.createOrReplaceTempView("transactions")

    # show all columns for 10 first rows
    result = spark.sql("select * from transactions limit 10")
    msg = "all columns for 10 first rows"
    show_query_result(result, msg, logger)

    # check how many fraud vs geniune transactions
    result = spark.sql("select is_fraud, count(*) as count from transactions group by is_fraud")
    msg = "how many fraud vs geniune transactions"
    show_query_result(result, msg, logger)

    # show distinct merchants and their categories and types which fraud is detected with expectation
    result = spark.sql("select distinct merchant_category, merchant_type, merchant, sum(case when high_risk_merchant = True then 1 else 0 end) as high_risk_count, sum(case when is_fraud = True then 1 else 0 end) as is_fraud_count from transactions group by merchant_category, merchant_type, merchant order by merchant_category, merchant_type, merchant, high_risk_count desc, is_fraud_count desc")
    msg = "merchants and their categories and types which fraud is detected with expectation"
    show_query_result(result, msg, logger)

    # check where frauds are detected
    result = spark.sql("select country, city_size, currency, sum(case when is_fraud = True then 1 else 0 end) as is_fraud_count from transactions group by country, city_size, currency order by is_fraud_count desc")
    msg = "places where frauds are detected"
    show_query_result(result, msg, logger)

    # check how much fraud amount is detected
    result = spark.sql("select country, currency, format_number(cast(sum(amount) as decimal(18,2)),2) as sum_amount from transactions where is_fraud = True group by country, currency order by sum(amount) desc")
    msg = "how much fraud amount is detected"
    show_query_result(result, msg, logger)

    # check on which devices and channels frauds are detected 
    result = spark.sql("select channel, device, count(*) as count from transactions where is_fraud = True group by channel, device order by channel, device, count desc")
    msg = "on which devices and channels frauds are detected "
    show_query_result(result, msg, logger)

    # check which customer_ids have the most fraudulant transactions
    result = spark.sql("select customer_id, count(*) as count from transactions where is_fraud = 1 group by customer_id order by count desc limit 10")
    msg = "customers with the most fraudulant transactions (limit 10)"
    show_query_result(result, msg, logger)


@show_elapsed_time
def prepare_data(spark, df, output_path, logger):
    # preparing the data and eliminate unused fields

    # split data-time column (timestamp)
    df_cleaned = df.withColumn("date", regexp_replace(date_format(col("timestamp"), "yyyy-MM-dd"),"-","").cast("integer"))
    df_cleaned = df_cleaned.withColumn("time_hour", date_format(col("timestamp"), "HH").cast("integer"))
    df_cleaned = df_cleaned.withColumn("time_minute", date_format(col("timestamp"), "mm").cast("integer"))
    df_cleaned = df_cleaned.withColumn("time_second", date_format(col("timestamp"), "ss").cast("integer")) 

    # change boolean columns to integer
    df_cleaned = df_cleaned.withColumn("card_present", col("card_present").cast("integer"))
    df_cleaned = df_cleaned.withColumn("high_risk_merchant", col("high_risk_merchant").cast("integer"))
    df_cleaned = df_cleaned.withColumn("weekend_transaction", col("weekend_transaction").cast("integer"))
    df_cleaned = df_cleaned.withColumn("is_fraud", col("is_fraud").cast("integer"))

    # select only usable fields
    df_cleaned = df_cleaned.select("customer_id", "card_number", "date", "time_hour", "time_minute", "time_second", "merchant_category", "merchant_type", "merchant", "amount", "currency", "country", "city", "city_size", "card_type", "card_present", "device", "channel", "distance_from_home", "high_risk_merchant", "transaction_hour", "weekend_transaction", "is_fraud")

    # processed data schema
    msg = "The schema of data after preparation is"
    show_schema(df_cleaned, msg, logger)
    
    # save first 1000 rows of the cleaned data as a csv file
    df_cleaned.createOrReplaceTempView("transactions")
    result = spark.sql("select * from transactions limit 1000")
    result.coalesce(1).write.mode("overwrite").csv(path.join(output_path,"processed_data_csv_top1000"), header = True)
    
    return df_cleaned

@show_elapsed_time
def main():
    # check if two arguments has been passed to the program
    if len(sys.argv)<3:
        print("The program needs two parameters!")
        sys.exit(1)

    # retrieve arguments values
    input_path = sys.argv[1]
    output_path = sys.argv[2]

    # create a logger instance
    logger = Logger()

    # read data from the input file
    spark, df = read_data(input_path, logger)

    # check data quality and make a report
    check_data_quality(spark, df, logger)

    # prepare data for the analysis
    processed_data = prepare_data(spark, df, output_path, logger)

    # save the whole processed data into a new csv file for analysis
    processed_data.coalesce(1).write.mode("overwrite").csv(path.join(output_path,"processed_data_csv_full"), header = True)

    # Save log file
    logger.save_file(path.join(output_path,"log"),spark)

    # stop spark session
    spark.stop()

if __name__ == "__main__":
    main()