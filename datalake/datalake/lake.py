import zmq
from zmq import ZMQError
from multiprocessing import Process
import argparse
import pymongo
import ujson as json
import time


def lake_worker(address: str, db_conn_string: str, capacity: int, max_wait: float):
    """
    Wrapper script for Process target invocation. Creates and starts the LakeWorker.
    :param address: str The address of the zmq socket we are listening to.
    :param db_conn_string: str The connection string of the MongoDB instance we are saving to.
    :param capacity: int The maximum number of data dicts we can buffer before saving to db.
    :param max_wait: float The maximum time in seconds to wait after the last data to flush the
    buffer.
    :return: None.
    """
    worker = LakeWorker(address=address, db_conn_string=db_conn_string, capacity=capacity,
                        max_wait=max_wait)
    worker.work()


class LakeWorker:
    # TODO figure out how to terminate this nicely without losing data stored in the buffer.

    def __init__(self, address:str, db_conn_string: str, capacity: int, max_wait: float):
        self.context = zmq.Context()
        self.receiver = self.context.socket(zmq.PULL)
        self.receiver.connect('tcp://' + address)
        # create the database if it doesn't already exist (handled by pymongo)
        self.db = pymongo.MongoClient(db_conn_string).training_data
        self.buffer = dict()
        self.capacity = capacity
        self.max_wait = max_wait
        # NOTE this dict is never cleared so will cause crashes once it gets too big.
        self.index_state = dict()
        self.all_indexed = True

    def work(self):
        """
        Main work loop where data is read from the socket, buffered and then sent to the db.
        :return: None.
        """
        counter = 0
        last_received = time.time()
        while True:
            # read the data
            try:
                msg = self.receiver.recv_string(flags=1)
                # print("received")
                last_received = time.time()
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
                    counter = 0
                # If its a new training run start tracking the index state.
                # set global indexed state as false.
                if col not in self.index_state:
                    self.index_state[col] = False
                    self.all_indexed = False
            except ZMQError as e:
                # print("Time passed: ",time.time() - last_received)  debugging...
                # if we don't receive anything for a while flush the buffer anyway.
                if (time.time() - last_received) > self.max_wait:
                    self.flush_buffer()
                    print("Flush database")
                    # reset counters so we flush at most every max_wait seconds.
                    last_received = time.time()
                    counter = 0
                    # use this opportunity to check indices and create them if not created.
                    if not self.all_indexed:
                        # create index on those that aren't indexed already and set index created.
                        self.index_collections()
                # print(e)

    def flush_buffer(self) -> None:
        """
        Sends all data stored in the buffer to the db and clears the buffer.
        :return: None
        """
        for col, data in self.buffer.items():
            result = self.db[col].insert_many(data)
            # print("Inserted to database")
        self.buffer = dict()

    def index_collections(self) -> None:
        for collection, indexed in self.index_state:
            if not indexed:
                # TODO verify.
                self.db[collection].create_index('total_minibatch', partialFilterExpression={
                    'total_minibatch': {'$exists': True}})
                self.index_state[collection] = True
        self.all_indexed = True


class LakeCoordinator:

    def __init__(self, source_address: str, max_processes: int,
                 db: str = "mongodb://localhost:27017/", capacity: int = 100, max_wait: float =
                 5) -> None:
        self.source_address = source_address
        self.max_processes = max_processes
        self.processes = list()
        self.db_conn_string = db
        self.max_wait = max_wait
        # max number of data dicts each process can store. Highly dependant on available RAM and
        # data size
        self.process_capacity = capacity // self.max_processes

    def start_lake(self) -> None:
        """
        Starts all the processes.
        :return: None
        """
        for i in range(self.max_processes):
            p = Process(target=lake_worker, args=(self.source_address, self.db_conn_string, self.process_capacity, self.max_wait), daemon=True)
            p.start()
            self.processes.append(p)

    def end_lake(self) -> None:
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
    parser.add_argument('--max_wait', type=float, help="The max time (in seconds) to wait after "
                                                       "last data received  before writing "
                                                       "current buffer to MongoDB. Default is 5s.",
                                                       default=5)
    args = parser.parse_args()
    lake = LakeCoordinator(source_address=args.address, max_processes=args.processes,
                           capacity=args.capacity, max_wait=args.max_wait)
    lake.start_lake()
    while True:
        line = input('Type kill to end reading.')
        if line == 'kill':
            lake.end_lake()
            exit(code=1)
