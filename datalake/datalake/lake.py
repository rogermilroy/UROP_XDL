import zmq
from multiprocessing import Process
import argparse
import sys


def lake_worker(address: str):
    context = zmq.Context()
    receiver = context.socket(zmq.PULL)
    receiver.connect('tcp://' + address)

    # TODO allow for kill signal.
    while True:
        msg = receiver.recv()
        print(msg)


class LakeCoordinator:

    def __init__(self, source_address: str, max_processes: int):
        self.source_address = source_address
        self.max_processes = max_processes
        self.processes = list()
        # TODO setup MongoDB connection.

    def start_lake(self):
        """
        Starts all the processes.
        :return: None
        """
        for i in range(self.max_processes):
            p = Process(target=lake_worker, args=self.source_address)
            self.processes.append(p)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create a datalake with some processes, "
                                                 "reading from an address.")
    parser.add_argument('address', help='The address of the server distributing the data.')
    parser.add_argument('processes', help='Number of processes wanted.')
    args = parser.parse_args()
    lake = LakeCoordinator(args.address, int(args.processes))
    lake.start_lake()
    while True:
        line = input('Type kill to end reading.')
        if line == 'kill':
            exit(code=1)
