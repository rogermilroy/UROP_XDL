import datalake.lake as lake
import zmq
import pymongo
import ujson as json
import argparse


def receive(address: str, db_conn_string: str) -> None:
    context = zmq.Context()
    # Create and connect receiving socket.
    receiver = context.socket(zmq.PULL)
    receiver.connect('tcp://' + address)
    poller = zmq.Poller()
    poller.register(receiver, zmq.POLLIN)
    db = pymongo.MongoClient(db_conn_string).training_data
    while True:
        socks = dict(poller.poll())

        # check if there is a message for receiver.
        if socks.get(receiver) == zmq.POLLIN:
            msg = receiver.recv_string()
            # print("received")
            # keep track of the time
            data = json.loads(msg)
            # create the identity string (becomes the name of the collection)
            col = data["training_run"]
            db[col].insert_one(data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create a datalake with some processes, "
                                                 "reading from an address.")
    parser.add_argument('address', type=str, help='The address of the server distributing the '
                                                  'data.')
    # parser.add_argument('--indices' ) TODO up to multiple inputs.
    args = parser.parse_args()
    receive(address=args.address, db_conn_string="mongodb://localhost:27017/")
