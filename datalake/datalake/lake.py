import zmq
from zmq import ZMQError
from multiprocessing import Process
import argparse
import pymongo
import ujson as json
import time


def lake_worker(address: str, coordinator: str, db_conn_string: str, capacity: int,
                max_wait: float) -> None:
    """
    Wrapper script for Process target invocation. Creates and starts the LakeWorker.
    :param address: str The address of the zmq socket we are listening to.
    :param coordinator: str The connection string for the LakeCoordinator for signalling.
    :param db_conn_string: str The connection string of the MongoDB instance we are saving to.
    :param capacity: int The maximum number of data dicts we can buffer before saving to db.
    :param max_wait: float The maximum time in seconds to wait after the last data to flush the
    buffer.
    :return: None.
    """
    worker = LakeWorker(address=address, coordinator=coordinator,
                        db_conn_string=db_conn_string, capacity=capacity, max_wait=max_wait)
    worker.work()


class LakeWorker:

    def __init__(self, address: str, coordinator: str, db_conn_string: str,
                 capacity: int, max_wait: float) -> None:
        # Create zmq context.
        self.context = zmq.Context()
        # Create and connect receiving socket.
        self.receiver = self.context.socket(zmq.PULL)
        self.receiver.connect('tcp://' + address)
        # Create and connect coordinator socket.
        self.coordinator = self.context.socket(zmq.SUB)
        self.coordinator.connect('ipc://' + coordinator)
        self.coordinator.setsockopt(zmq.SUBSCRIBE, b'')  # subscribe to everything.
        # Create a poller and register the sockets.
        self.poller = zmq.Poller()
        self.poller.register(self.receiver, zmq.POLLIN)
        self.poller.register(self.coordinator, zmq.POLLIN)
        # create the database if it doesn't already exist (handled by pymongo)
        self.db = pymongo.MongoClient(db_conn_string).training_data
        self.buffer = dict()
        self.capacity = capacity
        self.max_wait = max_wait
        # # NOTE this dict is never cleared so will cause crashes once it gets too big.
        # self.index_state = dict()
        # self.all_indexed = True

    def work(self):
        """
        Main work loop where data is read from the socket, buffered and then sent to the db.
        :return: None.
        """
        counter = 0
        last_received = time.time()
        while True:
            # read the data (non-blocking)
            socks = dict(self.poller.poll())

            # check if there is a message for receiver.
            if socks.get(self.receiver) == zmq.POLLIN:
                msg = self.receiver.recv_string()
                # print("received")
                # keep track of the time
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

            # if we haven't received data we check for timeout for flushing the buffer.
            elif (time.time() - last_received) > self.max_wait:
                self.flush_buffer()
                print("Flush database")
                # reset counters so we flush at most every max_wait seconds.
                last_received = time.time()
                counter = 0

            # check if there is a message from the coordinator.
            if socks.get(self.coordinator) == zmq.POLLIN:
                signal = self.coordinator.recv_string()
                if signal == "PAUSE":
                    # flush the buffer then do nothing until restarted.
                    print("Pausing work.")
                    self.flush_buffer()
                    self.pause_work()
                elif signal == "KILL":
                    print("Killing process")
                    self.flush_buffer()
                    # exit the process.
                    exit(code=1)

    def flush_buffer(self) -> None:
        """
        Sends all data stored in the buffer to the db and clears the buffer.
        :return: None
        """
        for col, data in self.buffer.items():
            result = self.db[col].insert_many(data)
            # print("Inserted to database")
        self.buffer = dict()

    def pause_work(self) -> None:
        # listen for a signal from the coordinator (blocking intentionally)
        signal = self.coordinator.recv_string()
        if signal == "RESTART":
            print("Restarting")
            self.work()
        elif signal == "KILL":
            # exit immediately as we have already flushed the buffer.
            exit(code=1)


class LakeCoordinator:

    def __init__(self, source_address: str, max_processes: int,
                 db: str = "mongodb://localhost:27017/", capacity: int = 100,
                 max_wait: float = 5.) -> None:
        self.source_address = source_address
        self.max_processes = max_processes
        self.processes = list()
        self.db_conn_string = db
        self.max_wait = max_wait
        # max number of data dicts each process can store. Highly dependant on available RAM and
        # data size
        self.process_capacity = capacity // self.max_processes
        # setup zmq communications with workers.
        self.context = zmq.Context()
        self.ipc_address = '/tmp/coordinate'
        self.worker_control = self.context.socket(zmq.PUB)
        self.worker_control.bind('ipc://' + self.ipc_address)

    def start_lake(self) -> None:
        """
        Starts all the processes.
        :return: None
        """
        for i in range(self.max_processes):
            p = Process(target=lake_worker,
                        args=(self.source_address, self.ipc_address, self.db_conn_string,
                              self.process_capacity, self.max_wait),
                        daemon=True)
            p.start()
            self.processes.append(p)

    def end_lake(self, timeout: float = 5.) -> None:
        """
        Stops all the processes. Attempts to stop them gracefully first. Waits up to timeout time
        and then terminates any remaining worker processes.
        :return: None
        """
        # attempt to exit gracefully
        self.worker_control.send_string("KILL")
        start_time = time.time()
        # allow some time to exit and check.
        while time.time() - start_time < timeout :
            for p in self.processes:
                if p.exitcode is not None:
                    self.processes.remove(p)
            # if they have all exited we have nothing left to do.
            if len(self.processes) > 0:
                print("All workers exited cleanly :)")
                return
        # if they still haven't exited, terminate them a little less politely.
        for p in self.processes:
            p.terminate()
            p.join()
        # after all the processes have been terminated we empty the list of processes.
        print("We had to be less polite and terminate some processes.")
        self.processes = list()

    def pause_lake(self) -> None:
        """
        Sends a signal to pause the worker processes.
        :return: None
        """
        self.worker_control.send_string("PAUSE")

    def restart_lake(self) -> None:
        """
        Sends a signal to restart the worker processes.
        :return: None
        """
        self.worker_control.send_string("RESTART")

    def add_index(self, collection: str, key_and_order: tuple, filter_expression: dict,
                  name: str) -> bool:
        """
        Adds a specified index to the database and returns whether it was successful or not.
        Side effect is that it pauses the reading of data.
        :param collection:
        :param key_and_order:
        :param filter_expression:
        :param name:
        :return: bool True if successful False otherwise
        """
        # pause the workers and wait for them to flush.
        self.pause_lake()
        time.sleep(1.)
        with pymongo.MongoClient(self.db_conn_string).training_data as db:
            # try to create the index
            db[collection].create_index(key_and_order, partialFilterExpression=filter_expression,
                                        name=name)
            # check if it was created, restart regardless
            if name not in db[collection].index_information():
                self.restart_lake()
                return False
            else:
                self.restart_lake()
                return True


def main():
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
                        default=5.)
    # parser.add_argument('--indices' ) TODO up to multiple inputs.
    args = parser.parse_args()
    lake = LakeCoordinator(source_address=args.address, max_processes=args.processes,
                           capacity=args.capacity, max_wait=args.max_wait)
    lake.start_lake()
    while True:
        line = input('Type kill to end reading.')
        if line == 'kill':
            lake.end_lake()
            exit(code=1)


if __name__ == '__main__':
    main()
