from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from concurrent import futures

import time
import os

import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite

import grpc
from model_send_pb2_grpc import FileServerServicer, add_FileServerServicer_to_server
from model_send_pb2 import Reply

def save_chunks_to_file(chunks):
    chunk = next(chunks)
    name = chunk.name
    print(name)
    with open(name + ".tflite", "wb") as f:
        f.write(chunk.buffer)
        for chunk in chunks:
            f.write(chunk.buffer)
    return name + ".tflite"

class Servicer(FileServerServicer):
    def __init__(self):
        pass

    def SendModel(self, request_iterator, context):
        print("receive")
        model_name = save_chunks_to_file(request_iterator)
        print("wait")
        accuracy, latency = inference(model_name)
        print(model_name)
        os.remove(model_name)        

        return Reply(accuracy=accuracy, latency=latency)

def load_labels(filename):
  with open(filename, 'r') as f:
    return [line.strip() for line in f.readlines()]

labels = load_labels(os.path.join( '..','labels.txt'))

# os.path.abspath(__file__), '..', 

def inference(model_name):
  interpreter = tflite.Interpreter(
    model_path=os.path.join( '.', model_name))
  interpreter.allocate_tensors()

  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()

  # check the type of the input tensor
  floating_model = input_details[0]['dtype'] == np.float32

  # NxHxWxC, H:1, W:2
  height = input_details[0]['shape'][1]
  width = input_details[0]['shape'][2]
  img = Image.open(os.path.join('..','grace_hopper.jpg')).resize((width, height))

  # add N dim
  input_data = np.expand_dims(img, axis=0)

  if floating_model:
    input_data = (np.float32(input_data) - args.input_mean) / args.input_std

  interpreter.set_tensor(input_details[0]['index'], input_data)

  start_time = time.time()
  interpreter.invoke()
  stop_time = time.time()

  output_data = interpreter.get_tensor(output_details[0]['index'])
  results = np.squeeze(output_data)

  top_k = results.argsort()[-5:][::-1]
  for i in top_k:
    if floating_model:
      print('{:08.6f}: {}'.format(float(results[i]), labels[i]))
    else:
      print('{:08.6f}: {}'.format(float(results[i] / 255.0), labels[i]))

  print('time: {:.3f}ms'.format((stop_time - start_time) * 1000))

  return 80.22, 4.2

if __name__ == '__main__':
  server_file = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
  add_FileServerServicer_to_server(
      Servicer(),
      server_file
  )
  server_file.add_insecure_port(f'[::]:{50000}')
  server_file.start()

  try:
    while True:
        time.sleep(60*60*24)
  except KeyboardInterrupt:
        server_file.stop(0) 
