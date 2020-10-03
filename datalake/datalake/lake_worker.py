import time

import ujson as json
import zmq
from datalake.database_interfaces import MongoInterface


def lake_worker(address: str, coordinator: str, feedback: str, db_conn_string: str, capacity: int,
                max_wait: float) -> None:
    """
    Wrapper script for Process target invocation. Creates and starts the LakeWorker.
    :param address: str The address of the zmq socket we are listening to.
    :param coordinator: str The connection string for the LakeCoordinator for signalling.
    :param feedback: str The connection string to provide feedback to coordinator.
    :param db_conn_string: str The connection string of the MongoDB instance we are saving to.
    :param capacity: int The maximum number of data dicts we can buffer before saving to db.
    :param max_wait: float The maximum time in seconds to wait after the last data to flush the
    buffer.
    :return: None.
    """
    worker = LakeWorker(address=address, coordinator=coordinator,
                        db_conn_string=db_conn_string, capacity=capacity, max_wait=max_wait,
                        feedback=feedback)
    worker.work()


class LakeWorker:

    def __init__(self, address: str, coordinator: str, feedback: str, db_conn_string: str,
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
        self.feedback = self.context.socket(zmq.PUB)
        self.feedback.bind('ipc://' + feedback)
        # Create a poller and register the sockets.
        self.poller = zmq.Poller()
        self.poller.register(self.receiver, zmq.POLLIN)
        self.poller.register(self.coordinator, zmq.POLLIN)
        # create the database if it doesn't already exist (handled by pymongo)
        self.db = MongoInterface(db_conn_string)
        self.buffer = dict()
        self.capacity = capacity
        self.max_wait = max_wait
        self.collections = list()
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
            # read the data (1 second timeout)
            socks = dict(self.poller.poll(1000))

            # check if there is a message for receiver.
            if self.receiver in socks and socks.get(self.receiver) == zmq.POLLIN:
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
                    self.collections.append(col)
                    # send the collection to coordinator for adding indices
                    self.feedback.send_string(col)
                else:
                    self.buffer[col].append(data)
                # We have a limit to how much is stored in the buffer and once it is reached we send
                # all the data to the db.
                counter += 1
                if counter == self.capacity:
                    self.flush_buffer()
                    counter = 0

            # check if there is a message from the coordinator.
            if self.coordinator in socks and socks.get(self.coordinator) == zmq.POLLIN:
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

            # if we haven't received data we check for timeout for flushing the buffer.
            if (time.time() - last_received) > self.max_wait:
                self.flush_buffer()
                print("Flush database")
                # reset counters so we flush at most every max_wait seconds.
                last_received = time.time()
                counter = 0

    def flush_buffer(self) -> None:
        """
        Sends all data stored in the buffer to the db and clears the buffer.
        :return: None
        """
        for col, data in self.buffer.items():
            self.db.insert_many(data, collection=col)

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
