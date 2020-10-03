import argparse
import time
from multiprocessing import Process

import pymongo
import zmq
from datalake.lake_worker import lake_worker


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
        self.worker_feedback = self.context.socket(zmq.SUB)
        self.worker_feedback.setsockopt(zmq.SUBSCRIBE, b'')  # subscribe to everything.

    def start_lake(self) -> None:
        """
        Starts all the processes.
        :return: None
        """
        for i in range(self.max_processes):
            p = Process(target=lake_worker,
                        args=(self.source_address, self.ipc_address, self.ipc_address + '-' + str(i),
                              self.db_conn_string, self.process_capacity, self.max_wait),
                        daemon=True)
            p.start()
            self.processes.append(p)
            self.worker_feedback.connect('ipc://' + self.ipc_address + '-' + str(i))

    def end_lake(self, timeout: float = 5.) -> None:
        """
        Stops all the processes. Attempts to stop them gracefully first. Waits up to timeout time
        and then terminates any remaining worker processes.
        :return: None
        """
        # attempt to exit gracefully
        self.worker_control.send_string("KILL")
        collections = set()
        # read all the message.
        while True:
            try:
                mess = self.worker_feedback.recv_string(flags=zmq.NOBLOCK)
                collections.add(mess)
            except:
                break

        for col in collections:
            self.add_index(collection=col, key_and_order=("total_minibatch", pymongo.DESCENDING),
                           filter_expression={"total_minibatch": {"$exists": True}},
                           name="total_minibatch",
                           timeout=100)
            self.add_index(collection=col, key_and_order=("final_model_state", pymongo.ASCENDING),
                           filter_expression={"final_model_state": {"$exists": True}},
                           name="final_model_state")

        start_time = time.time()
        # allow some time to exit and check.
        while time.time() - start_time < timeout:
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
                  name: str, timeout: int = 10) -> bool:
        """
        Adds a specified index to the database and returns whether it was successful or not.
        Side effect is that it pauses the reading of data.
        :param collection:
        :param key_and_order:
        :param filter_expression:
        :param name:
        :param timeout: optional if specified gives a maximum time before restarting the lake.
        :return: bool True if successful False otherwise
        """
        db = pymongo.MongoClient(self.db_conn_string).training_data
        # try to create the index
        db[collection].create_index([key_and_order], partialFilterExpression=filter_expression,
                                    name=name)
        # check if it was created
        start = time.time()
        while name not in db[collection].index_information():
            if (time.time() - start) > timeout:
                return False
        return True


def main():
    parser = argparse.ArgumentParser(description="Create a datalake with some processes, "
                                                 "reading from an address.")
    parser.add_argument('address', type=str, help='The address of the server distributing the '
                                                  'data as a string.')
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
        if line == 'pause':
            lake.pause_lake()
        if line == 'restart':
            lake.restart_lake()


if __name__ == '__main__':
    main()
