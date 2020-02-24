# -*- coding:utf-8 -*-
# @version: 1.0
# @author: wuxikun
# @date: '2020/2/19'


from concurrent import futures
import time
import grpc
import helloworld_pb2_grpc
import helloworld_pb2


class Greeter(helloworld_pb2_grpc.GreeterServicer):

    def SayHello(self, request, context):
        return helloworld_pb2.HelloReply(message='hello {msg}'.format(msg = request.name))

    def SayHelloAgain(self, request, context):
        return helloworld_pb2.HelloReply(message='hello {msg}'.format(msg=request.name))


def service():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    helloworld_pb2_grpc.add_GreeterServicer_to_server(Greeter(), server)

    server.add_insecure_port('[::]:50001')
    server.start()

    try:
        while True:
            time.sleep(60 * 60 * 24)  # one day in seconds
    except KeyboardInterrupt:
        server.stop(0)

if __name__ == '__main__':
    service()
