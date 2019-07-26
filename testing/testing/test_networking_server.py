#   Credit zmq guide zguide.zeromq.org.
#
#   Hello World server in Python
#   Binds REP socket to tcp://*:5555
#   Expects b"Hello" from client, replies with b"World"
#

import time
import zmq
from utils.network_details import get_ip

context = zmq.Context()
socket = context.socket(zmq.REP)
ip_addr = get_ip()
socket.bind("tcp://" + ip_addr + ":5555")

while True:
    #  Wait for next request from client
    message = socket.recv()
    print("Received request: %s" % message)

    #  Do some 'work'
    time.sleep(1)

    #  Send reply back to client
    socket.send(b"World")
