from pyspark.sql import SparkSession, Row
import sys
from os import path
from general import *

@show_elapsed_time
def read_data(input_path, logger : Logger):
    # open spark session
    spark = SparkSession.builder \
            .appName("Data Preparation MapReduce") \
            .config("spark.driver.memory", "4g") \
            .config("spark.executor.memory", "4g") \
            .getOrCreate()

    # read data from csv file
    df = spark.read.csv(input_path, inferSchema = True, header = True)

    # show schema of original data
    msg = "The schema of data before preparation is"
    show_schema(df, msg, logger)

    # convert dataframe to RDD for mapreduce operations
    rdd = df.rdd

    return spark, rdd

def show_schema(df, msg : str, logger:Logger):
    # show data schema and log it
    print(msg)
    content = capture_terminal_output(df.printSchema)
    logger.log(msg, content)

def map_reduce(rdd, map_function, msg : str, logger : Logger, limit : int = -1, has_two_output : bool = False):
    # map reduce operation
    mapped_rdd = rdd.map(map_function)

    # reduce the mapped data based on keys
    if not has_two_output:
        # when each key has one value
        reduced_rdd = mapped_rdd.reduceByKey(lambda x, y: x + y)
    else:
        # when each key has a pair of values
        reduced_rdd = mapped_rdd.reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1]))

    # sort the categories by key in ascending order
    sorted_rdd = reduced_rdd.sortBy(lambda x: x[0], ascending=True)

    # gather up the results
    if limit > 0:
        # if limit is specified use take for better performance
        result = sorted_rdd.take(limit)
    else:
        # collect all records in result
        result = sorted_rdd.collect()

    # print the result
    content = ""
    print(msg)
    for title, count in result:
        # check value or one of the pair of values greater than zero
        if count>0 if not has_two_output else count[0]>0 or count[1]>0:
            content += f"{title}: {count}\n"
            print(f"{title}: {count}")
    
    # log the result
    logger.log(msg, content)

@show_elapsed_time
def check_data_quality(spark, rdd, logger:Logger):
    # check data quality

    # check how many fraud vs geniune transactions
    msg = "how many fraud vs geniune transactions"
    map_reduce(rdd, lambda record: ("fraud", 1) if record["is_fraud"] == True else ("genuine", 1), msg, logger)
    
    # show distinct merchants and their categories and types which fraud is detected with expectation
    msg = "merchants and their categories and types which fraud is detected with expectation"
    map_reduce(rdd, lambda record: ("-".join([record["merchant_category"], record["merchant_type"], record["merchant"]]), (1 if record["high_risk_merchant"] == True else 0, 1 if record["is_fraud"] == True else 0)) , msg, logger, has_two_output=True)

    # check where frauds are detected
    msg = "places where frauds are detected"
    map_reduce(rdd, lambda record: ("-".join([record["country"], record["city_size"], record["currency"]]), 1 if record["is_fraud"] == True else 0) , msg, logger)

    # check how much fraud amount is detected
    msg = "how much fraud amount is detected "
    map_reduce(rdd, lambda record: ("-".join([record["country"], record["city_size"], record["currency"]]), record["amount"] if record["is_fraud"] == True else 0) , msg, logger)

    # check on which devices and channels frauds are detected 
    msg = "on which devices and channels frauds are detected"
    map_reduce(rdd, lambda record: ("-".join([record["channel"], record["device"]]), 1 if record["is_fraud"] == True else 0) , msg, logger)

    # check which customer_ids have fraudulant transactions
    msg = "customers with the most fraudulant transactions (limit 10)"
    map_reduce(rdd, lambda record: (record["customer_id"], 1 if record["is_fraud"] == True else 0) , msg, logger, limit=10)


@show_elapsed_time
def prepare_data(spark, rdd, output_path, logger):
    # preparing data and eliminate unused fields

    # split data-time column (timestamp)
    def split_timestamp(row):
        row_dict = row.asDict()
        timestamp = row_dict["timestamp"]
        row_dict["date"] = timestamp.year*10000+timestamp.month*100+timestamp.day
        row_dict["time_hour"] = timestamp.hour
        row_dict["time_minute"] = timestamp.minute
        row_dict["time_second"] = timestamp.second

        return Row(**row_dict)


    # change boolean columns to integer
    def cast_bool_int(row):
        row_dict = row.asDict()
        row_dict["card_present"] = 1 if row_dict["card_present"] == True else 0
        row_dict["high_risk_merchant"] = 1 if row_dict["high_risk_merchant"] == True else 0
        row_dict["weekend_transaction"] = 1 if row_dict["weekend_transaction"] == True else 0
        row_dict["is_fraud"] = 1 if row_dict["is_fraud"] == True else 0

        return Row(**row_dict)

    # apply transformations
    rdd = rdd.map(split_timestamp).map(cast_bool_int)

    # select only usable fields
    fields = ["customer_id", "card_number", "date", "time_hour", "time_minute", "time_second", "merchant_category", "merchant_type", "merchant", "amount", "currency", "country", "city", "city_size", "card_type", "card_present", "device", "channel", "distance_from_home", "high_risk_merchant", "transaction_hour", "weekend_transaction", "is_fraud"]
    rdd_cleaned = rdd.map(lambda row: Row(**{field: row[field] for field in fields}))

    # convert back to DataFrame for printing schema and writing to file
    df_cleaned = spark.createDataFrame(rdd_cleaned)

    # cleaned data schema
    msg = "The schema of data after preparation is"
    show_schema(df_cleaned, msg, logger)

    # save the first 100 rows of cleaned data as a CSV
    df_cleaned_top1000 = df_cleaned.limit(1000)
    df_cleaned_top1000.write.mode("overwrite").csv(path.join(output_path, "processed_data_csv_top1000"), header=True)

    return rdd_cleaned

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
    spark, rdd = read_data(input_path, logger)

    # check data quality and make a report
    check_data_quality(spark, rdd, logger)

    # prepare data for the analysis
    prepared_rdd = prepare_data(spark, rdd, output_path, logger)

    # Save log file
    logger.save_file(path.join(output_path,"log"),spark)

    # stop spark session
    spark.stop()

if __name__ == "__main__":
    main()