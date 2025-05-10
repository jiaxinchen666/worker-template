import os
import json
from pinecone.grpc import PineconeGRPC
from clip_client import Client
from google.cloud import storage
from dotenv import load_dotenv
import requests
import logging

load_dotenv('/root/search_engine/env/.env')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
pc = PineconeGRPC(api_key=PINECONE_API_KEY)

PINECONE_INDEX_1024="ds-images-1024"
index_1024 = pc.Index(PINECONE_INDEX_1024)

c_g = Client('grpc://0.0.0.0:51000')

OPENAI_URL="http://localhost:7000/openai/generate"

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/root/search_engine/env/gcp.json"
client = storage.Client()
bucket_name = 'tgaigc'  # 替换为您的存储桶名称
bucket = client.bucket(bucket_name)


def gen_img_prompt(character_content, user_content, character_name):
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json'
    }

    response = requests.post(OPENAI_URL, headers=headers, 
        data=json.dumps({"character_content": character_content, 
            "user_content": user_content,
            "character_name": character_name}))

    if response.status_code == 200:
        return response.json()
    else:
        logging.error(f"Request failed with status code {response.status_code}")
        return "Error"



    response_dict = {"img_prompt":[], "stage":[]}
    for r in range(int(llm_batch)):
        llm_response = gen_img_prompt(character_content, user_content, role)
        img_prompt_json = json.loads(llm_response)
        word_to_index = {
                'flirting': 0,
                'outercourse': 1,
                'penetration': 2
            }
        state = img_prompt_json["state"]
        stage = word_to_index[state]
        img_prompt = img_prompt_json["words"]
        response_dict["img_prompt"].append(img_prompt)
        response_dict["stage"].append(stage)    

    st.title("LLM IMG Prompt Output")
    st.session_state.search_df = pd.DataFrame(response_dict)


def prompt_embed(prompt: str, weights=None):
    parts = [p.strip() for p in prompt.split(',') if p.strip()]
    if not parts:
        raise ValueError("prompt 里没有有效短语")

    vecs = np.stack([c.encode([p])[0] for p in parts])  # shape = (n, d)

    if weights is None:
        weights = np.ones(len(parts))
    weights = np.asarray(weights, dtype=np.float32)
    if weights.shape[0] != vecs.shape[0]:
        raise ValueError("weights 数量必须等于短语数量")

    v = (weights[:, None] * vecs).sum(axis=0) / weights.sum()

    v /= np.linalg.norm(v) + 1e-9
    return v

# def encode_siglip(prompt):
#     before = time.time()
#     inp = proc(text=[prompt], return_tensors="pt").to(device)
#     with torch.no_grad():
#         t = model.get_text_features(**inp).squeeze(0)           # (1,D)
#     t = torch.nn.functional.normalize(t, dim=-1)
#     return t.cpu().numpy().astype("float32")

def encode_siglip(prompt):
    # important: make sure to set padding="max_length" as that's how the model was trained
    inputs = tokenizer([prompt], padding="max_length", return_tensors="pt")
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
    return text_features.cpu().numpy()

def encode_query_siglip(prompt):
    text_value_siglip = encode_siglip(prompt)
    result_siglip = index_1024.query(
        namespace=f"{role}_siglip_stage_{stage}",
        vector=text_value_siglip.tolist()[0],
        filter={
            "gcp_id": {"$nin": []},
        },
        top_k=topk,
        include_values=False,
        include_metadata=True
    )
    return result_siglip


def encode_query_vit(prompt):
    text_value_g = c_g.encode([prompt])
    text_value_L = c_L.encode([prompt])
    text_value_H = c_H.encode([prompt])
    text_value_jinav2 = jina_model.encode_text([prompt])

    result_L = index_768.query(
        namespace=f"{role}_stage_{stage}",
        vector=text_value_L.tolist()[0],
        filter={
            "gcp_id": {"$nin": []},
        },
        top_k=topk,
        include_values=False,
        include_metadata=True
    )

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

    result_H = index_1024.query(
        namespace=f"{role}_vitH_stage_{stage}",
        vector=text_value_H.tolist()[0],
        filter={
            "gcp_id": {"$nin": []},
        },
        top_k=topk,
        include_values=False,
        include_metadata=True
    )

    result_jina = index_1024.query(
        namespace=f"{role}_jinav2_stage_{stage}",
        vector=text_value_jinav2.tolist()[0],
        filter={
            "gcp_id": {"$nin": []},
        },
        top_k=topk,
        include_values=False,
        include_metadata=True
    )

    return result_L, result_g, result_H, result_jina    

if st.button("img_prompt_search"):
    topk = int(topk)
    for p_index, prompt in enumerate(prompt_list):
        init_time = time.time()
        result_L, result_g, result_H, result_jina = encode_query_vit(prompt)

        cols = st.columns(topk)
        for i, col in enumerate(cols):
            gcp_path=result_L["matches"][i]["metadata"]["gcp_id"]
            blob = bucket.blob(gcp_path)
            url = blob.generate_signed_url(
                expiration=datetime.timedelta(minutes=15),  # URL有效期为15分钟
                method='GET'
            )
            col.image(url)
            caption = f"Prompt {p_index}, vit-L: " + gcp_path.split("/")[-1]
            col.markdown(f"**{caption}**")

        cols = st.columns(topk)
        for i, col in enumerate(cols):
            gcp_path=result_g["matches"][i]["metadata"]["gcp_id"]
            blob = bucket.blob(gcp_path)
            url = blob.generate_signed_url(
                expiration=datetime.timedelta(minutes=15),  # URL有效期为15分钟
                method='GET'
            )
            col.image(url)
            caption = f"Prompt {p_index}, vit-g: " + gcp_path.split("/")[-1]
            col.markdown(f"**{caption}**")

        cols = st.columns(topk)
        for i, col in enumerate(cols):
            gcp_path=result_H["matches"][i]["metadata"]["gcp_id"]
            blob = bucket.blob(gcp_path)
            url = blob.generate_signed_url(
                expiration=datetime.timedelta(minutes=15),  # URL有效期为15分钟
                method='GET'
            )
            col.image(url)
            caption = f"Prompt {p_index}, vit-H: " + gcp_path.split("/")[-1]
            col.markdown(f"**{caption}**")
        
        cols = st.columns(topk)
        for i, col in enumerate(cols):
            gcp_path=result_jina["matches"][i]["metadata"]["gcp_id"]
            blob = bucket.blob(gcp_path)
            url = blob.generate_signed_url(
                expiration=datetime.timedelta(minutes=15),  # URL有效期为15分钟
                method='GET'
            )
            col.image(url)
            caption = f"Prompt {p_index}, jina_v2: " + gcp_path.split("/")[-1]
            col.markdown(f"**{caption}**")