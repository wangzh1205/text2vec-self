import os
import time

import requests
import torch
from pymilvus import connections, db, FieldSchema, DataType, CollectionSchema, Collection, MilvusClient
import numpy as np
import import_excel_data as excel_utils
from text2vec import SentenceModel


MILVUS_HOST = "localhost"
MILVUS_PORT = '19530'
DATA_BASE = 'migu_text'
MILVUS_COLLECTION_NAME = ('migu_bge_text_collection')
connection = connections.connect(user='minioadmin', password='minioadmin', host=MILVUS_HOST,
                                 port=MILVUS_PORT)
milvus_client = MilvusClient(uri=f"http://{MILVUS_HOST}:{MILVUS_PORT}", user='', password='', timeout=15, db_name=DATA_BASE)


def milvus_create_collection():
    dbs = db.list_database(timeout=15)
    print(dbs)
    if DATA_BASE not in dbs:
        database = db.create_database(DATA_BASE)
        print("generate database success database={}".format(DATA_BASE))
    db.using_database(DATA_BASE)
    fields = [
        FieldSchema(name="qid", dtype=DataType.INT64, description="pk", is_primary=True, auto_id=True),
        FieldSchema(name="question_type", dtype=DataType.VARCHAR, description="问题类型", max_length=512),
        FieldSchema(name="question_text", dtype=DataType.VARCHAR, description="问题问法", max_length=512),
        FieldSchema(name="answer", dtype=DataType.VARCHAR, description="问题答案", max_length=3072),
        FieldSchema(name="status", dtype=DataType.INT8, description="问题状态"),
        FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, description="向量值", dim=1024)
    ]
    schema = CollectionSchema(fields, "migu question milvus is the migu demo to introduce the APIs")
    text2vec_milvus = Collection(MILVUS_COLLECTION_NAME, schema)
    text2vec_milvus.flush()
    # text2vec_milvus.load()
    print("migu collection is created")


def create_milvus_index():
    db.using_database(DATA_BASE)
    text2vec_milvus = Collection(MILVUS_COLLECTION_NAME)
    # ---- 构建索引----
    index = {
        "index_type": "IVF_FLAT",
        "metric_type": "COSINE",  # COSINE、IP、L2
        "params": {"nlist": 1024},
    }
    text2vec_milvus.create_index("embeddings", index)
    print("index is created")


def add_data():
    with open('question-all.txt', 'r', encoding='utf-8') as rf:
        lines = rf.readlines()
        array = []
        for line in lines:
            try:
                questions = line.split('$')
                for i in range(len(questions)):
                    question = eval(questions[i])
                    db.using_database(DATA_BASE)
                    collection = Collection(MILVUS_COLLECTION_NAME)
                    embeddings = migu_embedding_req(question.get('question'))
                    if embeddings is not None:
                        question_dict = {
                            "question_type": question.get('source'),
                            "question_text": question.get('question'),
                            "answer": question.get('answer'),
                            "embeddings": embeddings,
                            "status": 1
                        }
                        array.append(question_dict)

                # print('array length:{}'.format(len(array)))
                if len(array) >= 500:
                    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                    print("date={}, insert size={}".format(current_time, len(array)))
                    insert_data(array)
                    array.clear()
            except Exception as e:
                write_request_fail_to_txt(line)
                print(f"Failed to parse data: {e}")


def add_question():
    db.using_database(DATA_BASE)
    collection = Collection(MILVUS_COLLECTION_NAME)
    excel_datas = excel_utils.read_excel(r'F:\pyworkspace\text2vec\web\migu_question.xlsx')
    array = []
    for data in excel_datas:
        embeddings = migu_embedding_req(data[1])
        if embeddings is not None:
            question_dict = {
                "question_type": data[0],
                "question_text": data[1],
                "answer": data[2],
                "embeddings": embeddings,
                "status": 1
            }
        array.append(question_dict)
    insert_data(array)


def insert_data(data):
    try:
        print("insert data size:{}".format(len(data)))
        milvus_client.insert(collection_name=MILVUS_COLLECTION_NAME, data=data)
        milvus_client.flush(MILVUS_COLLECTION_NAME)
        emb_collection = Collection(MILVUS_COLLECTION_NAME)
        emb_collection.load()
        print("Insertion successful.")
    except Exception as e:
        write_insert_fail_to_txt(data)
        print(f"Failed to insert data: {e}")


def migu_embedding_req(data: str):
    try:
        endpoint_url = 'http://36.139.59.28:37860/api/embedding'
        payload = {
            "texts": [data],
        }
        # HTTP headers for authorization
        headers = {
            "Content-Type": "application/json"
        }

        params = {
        }
        response = requests.post(endpoint_url, params=params, headers=headers, json=payload, timeout=(5, 30))

        if response.status_code != 200:
            print('response status fail {}'.format(response))
            write_request_fail_to_txt(data)
            return None

        parsed_response = response.json()
        # check for errors
        if parsed_response["code"] != 0:
            print('response code fail {}'.format(parsed_response["code"]))
            write_request_fail_to_txt(data)
            return None

        embeddings = parsed_response["data"][0]
        return embeddings
    except Exception as e:
        write_request_fail_to_txt(data)
        print(f"Failed to insert data: {e}")

    return None


def write_insert_fail_to_txt(question: str):
    filename = "question_bge_insert_fail.txt"
    if os.path.exists(filename):
        # print("file exists")
        with open(filename, "a", encoding='UTF-8') as f:
            f.write(question)
            f.write('\n')
    else:
        with open(filename, "w", encoding='UTF-8') as f:
            f.write(question)


def write_request_fail_to_txt(question: str):
    filename = "question_bge_request_fail.txt"
    if os.path.exists(filename):
        # print("file exists")
        with open(filename, "a", encoding='UTF-8') as f:
            f.write(question)
            f.write('\n')
    else:
        with open(filename, "w", encoding='UTF-8') as f:
            f.write(question)


def search_data(codes, limit, score):
    db.using_database(DATA_BASE)
    collection = Collection(MILVUS_COLLECTION_NAME)
    collection.load()
    query_item = {
        "data": [codes],
        "anns_field": "embeddings",
        "param": {"metric_type": "COSINE", "params": {"nprobe": 10}, "offset": 0},
        "limit": limit,
        "output_fields": ['question_type', 'question_text', 'answer']
    }
    print(f"query_item: {query_item}")
    hit_result = collection.search(**query_item)
    hits = hit_result[0]
    result = []
    for hit in hits:
        if hit.distance > score:
            result.append(hit)

    return result


if __name__ == '__main__':
    milvus_create_collection()
    create_milvus_index()
    # add_data()
    add_question()
    print('exit')













