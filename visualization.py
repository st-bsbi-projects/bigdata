from pyspark.sql import SparkSession
from pyspark.sql.functions import col, substring, sum
from pyspark.sql.types import StringType
import matplotlib.pyplot as plt
import seaborn as sns
from dp_spark import *
from general import *

@show_elapsed_time
def open_session(input_path, output_path, logger : Logger, mode : int = 1):
    # start spark session and load data into a dataframe
    # mode = 1 --> start a spark session and load processed data from a csv file which has been provided before
    # mode = 2 --> call read_data an prepare_data function from dp_spark module and prepare the original data and use it

    if mode == 1:
        # open spark session
        spark = SparkSession.builder \
                .appName("Machine Learning") \
                .config("spark.driver.memory", "4g") \
                .config("spark.executor.memory", "4g") \
                .getOrCreate()

        # read data from csv file
        processed_data = spark.read.csv(input_path, inferSchema = True, header = True)

        # show schema of original data
        msg = "The schema of processed data is"
        show_schema(processed_data, msg, logger)

    elif mode == 2:
        # read data from the input file
        spark, df = read_data(input_path, logger)

        # prepare data for the analysis
        processed_data = prepare_data(spark, df, output_path, logger)

    return spark, processed_data


def visualization(df, output_path : str):
    # create and save some visualizations

    fig = plt.figure(figsize=(16, 9), dpi=300)

    # select only needed columns to convert them to pandas df
    df_selected = df.select(["date", "time_hour", "merchant_type", "country", "channel","is_fraud"]).withColumn("year", substring(col("date").cast(StringType()),1,4)).withColumn("month", substring(col("date").cast(StringType()),5,2)).withColumn("day", substring(col("date").cast(StringType()),7,2))
    df_pandas = df_selected.toPandas()
    
    # countplot of fraud transaction based on day
    sns.countplot(df_pandas,x="day", hue="is_fraud")
    plt.savefig(fname=f"{output_path}/countplot_day.png", format="png", dpi = fig.dpi)
    plt.clf()
    # countplot of fraud transaction based on hour
    sns.countplot(df_pandas,x="time_hour", hue="is_fraud")
    plt.savefig(fname=f"{output_path}/countplot_hour.png", format="png", dpi = fig.dpi)
    plt.clf()
    # countplot of fraud transaction based on merchant_type
    sns.countplot(df_pandas,x="merchant_type", hue="is_fraud")
    plt.savefig(fname=f"{output_path}/countplot_merchant_type.png", format="png", dpi = fig.dpi)
    plt.clf()
    # countplot of fraud transaction based on country
    sns.countplot(df_pandas,x="country", hue="is_fraud")
    plt.savefig(fname=f"{output_path}/countplot_country.png", format="png", dpi = fig.dpi)
    plt.clf()
    # countplot of fraud transaction based on channel
    sns.countplot(df_pandas,x="channel", hue="is_fraud")
    plt.savefig(fname=f"{output_path}/countplot_channel.png", format="png", dpi = fig.dpi)
    plt.clf()

    # select only needed columns to convert them to pandas dataframe
    df_selected = df.groupBy(["country", "is_fraud"]).agg(sum("amount").alias("total_amount"))
    df_pandas = df_selected.toPandas()
    # scatter plot country and amount
    sns.barplot(df_pandas, x="country", y="total_amount",hue="is_fraud")
    legend = plt.legend()
    legend.set_title("is_fraud")
    legend.set_bbox_to_anchor((1,1))
    plt.savefig(fname=f"{output_path}/barplot_country_amount.png", format="png", dpi = fig.dpi)
    plt.clf()

    # select only needed columns to convert them to pandas df
    df_selected = df.groupBy(["merchant_category", "merchant_type", "is_fraud"]).agg(sum("amount").alias("total_amount"))
    df_pandas = df_selected.toPandas()
    # scatter plot merchant_category and merchant_type
    sns.scatterplot(df_pandas, x="merchant_category", y="merchant_type", hue="is_fraud", size="total_amount", sizes = (200,1000))
    legend = plt.legend()
    legend.set_title("legends")
    legend.set_bbox_to_anchor((1,1))
    plt.grid(visible=True)
    plt.savefig(fname=f"{output_path}/scatter_merchant_category_type_amountsize.png", format="png", dpi = fig.dpi)
    plt.clf()

    # select only needed columns to convert them to pandas df
    df_selected = df.groupBy(["currency", "channel", "is_fraud"]).agg(sum("amount").alias("total_amount"))
    df_pandas = df_selected.toPandas()
    # scatter plot currency and channel
    sns.scatterplot(df_pandas, x="currency", y="channel", hue="is_fraud", size="total_amount", sizes = (200,1000))
    legend = plt.legend()
    legend.set_title("legends")
    legend.set_bbox_to_anchor((1,1))
    plt.savefig(fname=f"{output_path}/scatter_currency_channel_amountsize.png", format="png", dpi = fig.dpi)
    plt.clf()

    # select only needed columns to convert them to pandas df
    df_selected = df.groupBy(["city_size", "card_type", "is_fraud"]).agg(sum("amount").alias("total_amount"))
    df_pandas = df_selected.toPandas()
    # scatter plot currency and channel
    sns.scatterplot(df_pandas, x="city_size", y="card_type", hue="is_fraud", size="total_amount", sizes = (200,1000))
    legend = plt.legend()
    legend.set_title("legends")
    legend.set_bbox_to_anchor((1,1))
    plt.savefig(fname=f"{output_path}/scatter_citysize_cardtype_amountsize.png", format="png", dpi = fig.dpi)
    plt.clf()

    numeric_cols = [col for col, dtype in df.dtypes if dtype in ["int", "bigint", "float", "double", "decimal"]]
    corr = df.select(numeric_cols).toPandas().corr()
    sns.heatmap(corr, cmap="coolwarm", annot=True)
    plt.savefig(fname=f"{output_path}/heatmap_correlation.png", format="png", dpi = fig.dpi)
    plt.clf()

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

    # open spark session and load clean data into processed_data
    spark, processed_data = open_session(input_path, output_path, logger, 1)

    visualization(processed_data, output_path)

    # Save log file
    logger.save_file(path.join(output_path,"log"),spark)

    # stop spark session
    spark.stop()

if __name__ == "__main__":
    main()

