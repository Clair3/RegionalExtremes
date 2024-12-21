import datetime


def int_or_none(value):
    if value.lower() == "none":
        return None
    return int(value)


class Logger:
    def __init__(self, filepath: str = "log.txt"):
        self.filepath = filepath / "log.txt"
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self.filepath.touch(exist_ok=True)

    def printt(self, message: str):
        """
        Small function to track the different steps of the program with the time.
        Prints to console and appends to a file.
        """
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted_message = f"[{current_time}] {message}"

        # Print to console
        print(formatted_message)

        # Append to file
        with open(self.filepath, "a") as file:
            file.write(formatted_message + "\n")


# Global variable to hold the logger instance
logger = None


def initialize_logger(filepath: str = "log.txt"):
    global logger
    logger = Logger(filepath)


def printt(message: str):
    if logger is None:
        raise RuntimeError("Logger not initialized. Call initialize_logger() first.")
    logger.printt(message)


# Not yet used
# def timing_decorator(func):
#     def wrapper(*args, **kwargs):
#         start_time = time.time()
#         result = func(*args, **kwargs)
#         end_time = time.time()
#         printt(f"Function {func.__name__} took {end_time - start_time:.4f} seconds")
#         return result
#
#     return wrapper
