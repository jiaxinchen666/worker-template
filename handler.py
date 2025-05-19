"""Example handler file."""

import runpod
import os
import json


def handler(job):
    """Handler function that will be used to process jobs."""
    job_input = job["input"]
    character_content = job_input.get("character_content", "")
    user_content = job_input.get("user_content", "")
    character_name = job_input.get("character_name", "")
    topk = job_input.get("topk", 1)

    
    return {"result": character_name}

runpod.serverless.start({"handler": handler})
