from __future__ import print_function
from typing import List

import logging
import uuid
import grpc
import json
import random
import base64
from pathlib import Path

import service_pb2
import service_pb2_grpc

def run(image_paths: List[Path], ids: str):
    print(image_paths)
    data = []
    for image_path in image_paths:
        with image_path.open("rb") as image_file:
            encoded_image_string = base64.b64encode(image_file.read()).decode("utf-8")
        data.append({"image": encoded_image_string, "id": ids})
    
    content = json.dumps(data).encode()
    
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = service_pb2_grpc.ServiceStub(channel)
        response = stub.Call(service_pb2.Request(Input=content, 
                                                 info=service_pb2.Info(FunctionName="MNIST", 
                                                                       Timeout="None", 
                                                                       Runtime="Python3", 
                                                                       Limits=service_pb2.Resources(Memory="None", 
                                                                                                    CPU="None", 
                                                                                                    GPU="None"), 
                                                                       Trigger=service_pb2.Trigger(Name="None", 
                                                                                                   Topic="None", 
                                                                                                   Time="None")
                                                                        )
                                                )
        )

    print("Service client received: " + response.Output)


if __name__ == "__main__":
    logging.basicConfig()
    images_path = list(Path("..").glob("**/*.png"))
    
    k = 2
    random_image_paths = []
    for i in range(k):        
        random_image_paths.append(random.choice(images_path))

    run(image_paths=random_image_paths, ids=str(uuid.uuid4()))