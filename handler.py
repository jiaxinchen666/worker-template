"""Example handler file."""

import runpod
import os
import json
from pinecone import Pinecone
from clip_client import Client
from google.cloud import storage
from dotenv import load_dotenv
import requests
import logging
from img_prompt import gen_img_prompt
import datetime

pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))

PINECONE_INDEX_1024="ds-images-1024"
index_1024 = pc.Index(PINECONE_INDEX_1024)

import time
import sys

import time
import sys

def wait_for_clip_server(host: str = 'localhost', port: int = 51000, timeout: int = 600):
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            c_g = Client('grpc://0.0.0.0:51000')
            _ = c_g.encode(["test"])  # 发送简单请求
            print(f"[✓] clip-server is ready at {host}:{port}")
            return
        except Exception as e:
            print(f"[...] Waiting for clip-server ({e})")
            time.sleep(10)
    print(f"[✗] Timeout: clip-server not available after {timeout} seconds.", file=sys.stderr)
    sys.exit(1)

# 在初始化 clip-client 之前调用
wait_for_clip_server()

# c_g = Client('grpc://0.0.0.0:51000')

json_path = "/tmp/gcp.json"
with open(json_path, "w") as f:
    f.write(os.getenv('GOOGLE_APPLICATION_CREDENTIALS'))

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = json_path
client = storage.Client()
bucket_name = 'tgaigc'  # 替换为您的存储桶名称
bucket = client.bucket(bucket_name)

def query_encode_search(prompt, topk, role, stage):
    logging.info(f"query_encode_search: {prompt}, {topk}, {role}, {stage}")
    text_value_g = c_g.encode([prompt])

    result_g = index_1024.query(
        namespace=f"{role}_vitg_stage_{stage}",
        vector=text_value_g.tolist()[0],
        filter={
            "gcp_id": {"$nin": []},
        },
        top_k=topk,
        include_values=False,
        include_metadata=True
    )

    return result_g  

def gen_view_url(result):
    url_list = []
    for i in range(len(result["matches"])):
        gcp_path=result["matches"][i]["metadata"]["gcp_id"]
        blob = bucket.blob(gcp_path)
        url = blob.generate_signed_url(
            expiration=datetime.timedelta(minutes=15),  # URL有效期为15分钟
            method='GET'
        )
        url_list.append(url)
    return url_list



def handler(job):
    """Handler function that will be used to process jobs."""
    job_input = job["input"]
    character_content = job_input.get("character_content", "")
    user_content = job_input.get("user_content", "")
    character_name = job_input.get("character_name", "")
    topk = job_input.get("topk", 1)

    img_prompt = gen_img_prompt(character_content, user_content, character_name)
    if img_prompt is None:
        return {"error": "Failed to generate image prompt"}

    result = query_encode_search(img_prompt["img_prompt"], topk, character_name, img_prompt["stage"])
    url_list = gen_view_url(result)
    return {"result": url_list}

runpod.serverless.start({"handler": handler})
