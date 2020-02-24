# -*- coding:utf-8 -*-
# @version: 1.0
# @author: wuxikun
# @date: '2020/2/19'

import grpc
import helloworld.helloworld_pb2_grpc as helloworld_pb2_grpc
import helloworld.helloworld_pb2 as helloworld_pb2


def run():
    # channel = grpc.insecure_channel('[::]:50001')
    channel = grpc.insecure_channel('localhost:50051')

    stub = helloworld_pb2_grpc.GreeterStub(channel)
    response = stub.SayHello(helloworld_pb2.HelloRequest(name="Alex"))
    # response = stub.SayHello(helloworld_pb2.HelloRequest(name='czl'))
    #
    print(response.message)

    response = stub.SayHelloAgain(helloworld_pb2.HelloRequest(name='Nancy'))
    # stub.SayHelloAgain
    print(response.message)

if __name__ == '__main__':
    run()