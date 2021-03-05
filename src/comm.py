import time
import grpc

from model_send_pb2_grpc import FileServerStub
from model_send_pb2 import Model

CHUNK_SIZE = 1024 * 1024 # 1MB

def get_file_chunks(filename, vid):
    """Get file chunks"""
    with open(filename, 'rb') as f:
        while True:
            piece = f.read(CHUNK_SIZE)
            if len(piece) == 0:
                return
            yield Model(name=vid, buffer=piece)


channel_n = grpc.insecure_channel("localhost:50000")
stub = FileServerStub(channel_n)

req = get_file_chunks('./mobilenet_v2_1.0_224_quant.tflite', str(time.time()))
res = stub.SendModel(req)

print(res.accuracy, res.latency)
