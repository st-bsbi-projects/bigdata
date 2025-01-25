from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, LinearSVC
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
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


def encode_categorical_columns(processed_data):
    # encode categorical columns and change them to indexed numbers
    categorical_columns = [
        ("merchant_category", "merchant_category_index"),
        ("merchant_type", "merchant_type_index"),
        ("merchant", "merchant_index"),
        ("currency", "currency_index"),
        ("country", "country_index"),
        ("city", "city_index"),
        ("city_size", "city_size_index"),
        ("card_type", "card_type_index"),
        ("device", "device_index"),
        ("channel", "channel_index"),
    ]

    # in each iteration one categorical column is encoded
    indexed_data = processed_data
    for input_col, output_col in categorical_columns:
        indexer = StringIndexer(inputCol=input_col, outputCol=output_col)
        model = indexer.fit(indexed_data)
        indexed_data = model.transform(indexed_data)

    return indexed_data

@show_elapsed_time
def classification(processed_data, logger : Logger):
    # prepare data for analysis then creare classifications model and train then for evaluation and comparison
    
    # encode categorical variables
    indexed_data = encode_categorical_columns(processed_data)

    # assemble all feature columns into a single vector named "features"
    assembler = VectorAssembler(inputCols=["date", "time_hour", "time_minute", "time_second", "merchant_category_index", "merchant_type_index", "merchant_index", "amount", "currency_index", "country_index", "city_index", "city_size_index", "card_type_index", "card_present", "device_index", "channel_index", "distance_from_home", "high_risk_merchant", "transaction_hour", "weekend_transaction"], outputCol="features")
    # assembler = VectorAssembler(inputCols=["time_hour", "amount", "card_present", "distance_from_home", "high_risk_merchant", "transaction_hour"], outputCol="features")

    # transform data using assembler
    transformed_data = assembler.transform(indexed_data)

    # select only features and label columns
    final_data = transformed_data.select("features", "is_fraud")

    evaluate_models(final_data, logger)


def evaluate_models(final_data, logger : Logger):
    # create a list of classifiers and their parameters
    classifiers = [
        LogisticRegression(featuresCol="features", labelCol="is_fraud", predictionCol="predicted_is_fraud"),
        LinearSVC(featuresCol="features", labelCol="is_fraud", predictionCol="predicted_is_fraud"),
        RandomForestClassifier(featuresCol="features", labelCol="is_fraud", predictionCol="predicted_is_fraud", maxBins=200)
    ]

    # split data into train and test sets
    train_data, test_data = final_data.randomSplit(weights=[0.8, 0.2], seed=42)

    for classifier in classifiers:
        # create and train the model for every classifier
        cl_model = classifier.fit(train_data)

        # predict test_data with the model
        pred_data = cl_model.transform(test_data)

        # print the name of classifier
        logger.log(f"{classifier.__class__.__name__} evaluation results","")

        # Area Under ROC metric
        BCE = BinaryClassificationEvaluator(rawPredictionCol="predicted_is_fraud", labelCol="is_fraud", metricName="areaUnderROC")
        areaUnderROC = BCE.evaluate(pred_data)
        logger.log("Area Under ROC", str(areaUnderROC))

        # Area Under PR metric
        BCE = BinaryClassificationEvaluator(rawPredictionCol="predicted_is_fraud", labelCol="is_fraud", metricName="areaUnderPR")
        areaUnderPR = BCE.evaluate(pred_data)
        logger.log("Area Under PR", str(areaUnderPR))

        # Accuracy metric
        MCE = MulticlassClassificationEvaluator(predictionCol="predicted_is_fraud", labelCol="is_fraud", metricName="accuracy")
        accuracy = MCE.evaluate(pred_data)
        logger.log("Accuracy", str(accuracy))

        # Weighted Precision metric
        MCE = MulticlassClassificationEvaluator(predictionCol="predicted_is_fraud", labelCol="is_fraud", metricName="weightedPrecision")
        weightedPrecision = MCE.evaluate(pred_data)
        logger.log("Weighted Precision", str(weightedPrecision))

        # Weighted Recall metric
        MCE = MulticlassClassificationEvaluator(predictionCol="predicted_is_fraud", labelCol="is_fraud", metricName="weightedRecall")
        weightedRecall = MCE.evaluate(pred_data)
        logger.log("Weighted Recall", str(weightedRecall))

        # F1 Score metric
        MCE = MulticlassClassificationEvaluator(predictionCol="predicted_is_fraud", labelCol="is_fraud", metricName="f1")
        f1 = MCE.evaluate(pred_data)
        logger.log("F1 Score", str(f1))

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

    # classification modeling
    classification(processed_data, logger)

    # Save log file
    logger.save_file(path.join(output_path,"log"),spark)

    # stop spark session
    spark.stop()

if __name__ == "__main__":
    main()

