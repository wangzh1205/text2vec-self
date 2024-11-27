from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,db
)
import random

connections.connect(host="127.0.0.1", port="19530",db_name='book')


def create_collection():
    # collections = connections.connect(host="127.0.0.1", port="19530", db_name="book")
    # print("collections : {}".format(collections))
    if utility.has_collection('hello_milvus'):
        print('collection is 存在')
    else:
        fields = [
            FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="random", dtype=DataType.DOUBLE),
            FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=8)
        ]
        schema = CollectionSchema(fields, "hello_milvus is the simplest demo to introduce the APIs")
        hello_milvus = Collection("hello_milvus", schema)


def add_data():
    entities = [
        [i for i in range(1)],  # field pk
        [float(random.randrange(-20, -10)) for _ in range(1)],  # field random
        [[random.random() for _ in range(8)] for _ in range(1)],  # field embeddings
    ]
    print("type={}, entities={}".format(type(entities), entities))
    e = [random.random() for _ in range(8)]
    print("type={}, e={}".format(type(e), e))
    s = [{"pk":10, "random":12, "embeddings":e}]
    print("type={}, s={}".format(type(s), s))
    hello_milvus = Collection('hello_milvus')
    insert_result = hello_milvus.insert(s)
    # After final entity is inserted, it is best to call flush to have no growing segments left in memory
    hello_milvus.flush()
    return entities


def create_index():
    index = {
        "index_type": "IVF_FLAT",
        "metric_type": "L2",
        "params": {"nlist": 128},
    }
    hello_milvus = Collection('hello_milvus')
    hello_milvus.create_index("embeddings", index)


def search(entities):
    hello_milvus = Collection('hello_milvus')
    hello_milvus.load()
    vectors_to_search = entities[-1][-2:]
    print("search type={}, data={}".format(type(vectors_to_search), vectors_to_search))
    search_params = {
        "metric_type": "L2",
        "params": {"nprobe": 10},
    }
    result = hello_milvus.search(vectors_to_search, "embeddings", search_params, limit=3, output_fields=["random"])
    return result


if __name__ == '__main__':
    create_collection()
    entities = add_data()
    print("type={}, entities={}".format(type(entities), entities))
    create_index()
    hit = search(entities)
    print("type={}, hit={}".format(type(hit), hit))
