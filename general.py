import time
from pyspark.sql import SparkSession
from pyspark.sql.types import StringType
import sys
from io import StringIO

class Logger:
    # create class variables
    logged = ""
    msg_counter : int
    def __init__(self):
        # class initialization
        self.logged = ""
        self.msg_counter = 0

    def log(self, msg : str, content : str):
        # set message counter
        self.msg_counter += 1
        # store msg title and its content
        self.logged += " ".join([str(self.msg_counter), msg, "\n"])
        if content != "":
            self.logged += "\n".join([content, ""]) 

    def save_file(self, folder_path : str, spark : SparkSession):
        # save text file based on the logged string
        logged_df = spark.createDataFrame([self.logged], StringType()).toDF("logs")
        logged_df.coalesce(1).write.mode("overwrite").text(folder_path)


def capture_terminal_output(func, *args, **kwargs):
    # capture whatever is passed to the terminal, because pyspark dataframe functions only pass well structured output to the terminal
    try:
        # redirect sys.stdout to capture output
        buffer = StringIO()
        old_stdout = sys.stdout
        # redirect stdout to StringIO buffer
        sys.stdout = buffer  

        # show as many rows as requested with passing proper parameters
        func(*args, **kwargs)
        # ensure all output is flushed to the buffer
        sys.stdout.flush()

        # retrieve the captured output as a string
        content = buffer.getvalue()

    finally:
        # restore the original stdout
        sys.stdout = old_stdout

    # print the content because we grab all of it from terminal
    print(content)
    
    return content

def is_float(num : str):
    # check if string is a float or not
    try:
        f_num = float(num)
    except:
        return False
    return True

def show_elapsed_time(original_func):
    # decorator function to define start and end of function and print elapsed time
    def wrapper_function(*args, **kwargs):
        print(f"\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>function {original_func.__name__} started\n")
        current_time = time.time()
        result = original_func(*args, **kwargs)
        elapsedtime = int(time.time() - current_time)
        print(f"\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>function {original_func.__name__} ended ==> Elapsed time: {elapsedtime} seconds\n")
        if result : return result

    return wrapper_function