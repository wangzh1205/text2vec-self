from pymilvus import connections, db, FieldSchema, DataType, CollectionSchema, Collection
import numpy as np


MILVUS_HOST = "localhost"
MILVUS_PORT = '19530'
DATA_BASE = 'text2vec'
connection = connections.connect(user='minioadmin', password='minioadmin', host=MILVUS_HOST,
                                 port=MILVUS_PORT)


def milvus_create_collection():
    dbs = db.list_database(timeout=15)
    print(dbs)
    if DATA_BASE not in dbs:
        database = db.create_database(DATA_BASE)
        print("create database success database={}".format(DATA_BASE))
    db.using_database("text2vec")
    fields = [
        FieldSchema(name="w_id", dtype=DataType.VARCHAR, max_length=32, description="pk", is_primary=True, auto_id=False),
        # FieldSchema(name="userid", dtype=DataType.STRING, max_length=64),
        FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, description="向量值", dim=768),
        FieldSchema(name="keywords", dtype=DataType.VARCHAR, max_length=1024, description="实际内容，后期关联至其他存储")
    ]
    schema = CollectionSchema(fields, "text2vec_milvus is the text2vec demo to introduce the APIs")
    text2vec_milvus = Collection("text2vec_milvus", schema)
    text2vec_milvus.flush()
    # text2vec_milvus.load()


def create_milvus_index():
    db.using_database(DATA_BASE)
    text2vec_milvus = Collection('text2vec_milvus')
    # ---- 构建索引----
    index = {
        "index_type": "IVF_FLAT",
        "metric_type": "COSINE",  # COSINE、IP、L2
        "params": {"nlist": 1024},
    }
    text2vec_milvus.create_index("embeddings", index)


def add_data():
    db.using_database(DATA_BASE)
    collection = Collection('text2vec_milvus')
    data = generate_data()
    mr = collection.insert([data[0], data[1], data[2]])
    # mr = collection.insert([data[1], data[2]])
    print(mr)
    collection.load()


def generate_data():
    w_id, embeddings, keyword = [], [], []
    data_num = 100
    for idx in range(0, data_num):
        w_id.append(str(idx))
        embeddings.append(np.random.normal(0, 1, 768).tolist())
        keyword.append(f'random num {idx}')
    return w_id, embeddings, keyword


def search_data(codes, limit):
    db.using_database('text2vec')
    collection = Collection('text2vec_milvus')
    collection.load()
    query_item = {
        "data": [codes],
        "anns_field": "embeddings",
        "param": {"metric_type": "COSINE", "params": {"nprobe": 10}, "offset": 0},
        "limit": limit,
        "output_fields": ['w_id', 'keywords']
    }
    print(f"query_item: {query_item}")
    result = collection.search(**query_item)
    return result


if __name__ == '__main__':
    milvus_create_collection()
    create_milvus_index()
    add_data()
    print('exit')













