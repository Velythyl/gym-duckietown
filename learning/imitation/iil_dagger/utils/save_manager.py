import datetime
import os


class SaveManager:
    def __init__(self, storage_location, graph_name):
        self.storage_location = storage_location + "/" + (
            graph_name + str(datetime.datetime.now()).replace(" ",
                                                              "") if graph_name is not None else datetime.datetime.now().strftime(
                "%Y%m%d-%H%M%S")
        )

    def __str__(self):
        return self.storage_location

    def get_filepath(self, filename):
        return os.getcwd() + "/" + self.storage_location + "/" + filename
