import zmq
from multiprocessing import Process
import argparse
import pymongo
import zmq.utils.jsonapi as json


def lake_worker(address: str, db_conn_string: str, capacity: int):
    """
    Wrapper script for Process target invocation. Creates and starts the LakeWorker.
    :param address: str The address of the zmq socket we are listening to.
    :param db_conn_string: str The connection string of the MongoDB instance we are saving to.
    :param capacity: int The maximum number of data dicts we can buffer before saving to db.
    :return: None.
    """
    worker = LakeWorker(address=address, db_conn_string=db_conn_string, capacity=capacity)
    worker.work()


class LakeWorker:
    # TODO figure out how to terminate this nicely without losing data stored in the buffer.

    def __init__(self, address:str, db_conn_string: str, capacity: int):
        self.context = zmq.Context()
        self.receiver = self.context.socket(zmq.PULL)
        self.receiver.connect('tcp://' + address)
        # create the database if it doesn't already exist (handled by pymongo)
        self.db = pymongo.MongoClient(db_conn_string).training_data
        self.buffer = dict()
        self.capacity = capacity

    def work(self):
        """
        Main work loop where data is read from the socket, buffered and then sent to the db.
        :return: None.
        """
        counter = 0
        while True:
            # read the data
            msg = self.receiver.recv()
            data = json.loads(msg)
            # create the identity string (becomes the name of the collection)
            col = data["model_name"] + str(data["training_run_number"])
            # check if we have already seen data from this collection. Add to list if yes,
            # create if not.
            if col not in self.buffer:
                self.buffer[col] = [data]
            else:
                self.buffer[col].append(data)
            # We have a limit to how much is stored in the buffer and once it is reached we send
            # all the data to the db.
            counter += 1
            if counter == self.capacity:
                self.flush_buffer()

    def flush_buffer(self):
        """
        Sends all data stored in the buffer to the db and clears the buffer.
        :return: None
        """
        for col, data in self.buffer.items():
            result = self.db.col.insert_many(data, bypass_document_validation=True)
        self.buffer = dict()


class LakeCoordinator:

    def __init__(self, source_address: str, max_processes: int,
                 db: str = "mongodb://localhost:27017/", capacity: int = 100):
        self.source_address = source_address
        self.max_processes = max_processes
        self.processes = list()
        self.db_conn_string = db
        # max number of data dicts each process can store. Highly dependant on available RAM and
        # data size
        self.process_capacity = capacity //self.max_processes

    def start_lake(self):
        """
        Starts all the processes.
        :return: None
        """
        for i in range(self.max_processes):
            p = Process(target=lake_worker, args=(self.source_address, self.db_conn_string, self.process_capacity), daemon=True)
            p.start()
            self.processes.append(p)

    def end_lake(self):
        """
        Stop all the processes.
        :return: None
        """
        # stop the processes
        for p in self.processes:
            p.terminate()
            p.join()
        # after all the processes have been terminated we empty the list of processes.
        self.processes = list()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create a datalake with some processes, "
                                                 "reading from an address.")
    parser.add_argument('address', type=str, help='The address of the server distributing the '
                                                  'data.')
    parser.add_argument('processes', type=int, help='Number of processes wanted.')
    parser.add_argument('capacity', type=int, help="The maximum amount of data dicts to be "
                                                   "buffered before saving to MongoDB. Attention! "
                                                   "This is highly dependant on the amount of RAM "
                                                   "available and the size of the network being "
                                                   "trained. Be cautious and test to find the "
                                                   "optimal amount.")
    args = parser.parse_args()
    lake = LakeCoordinator(source_address=args.address, max_processes=args.processes,
                           capacity=args.capacity)
    lake.start_lake()
    while True:
        line = input('Type kill to end reading.')
        if line == 'kill':
            lake.end_lake()
            exit(code=1)
