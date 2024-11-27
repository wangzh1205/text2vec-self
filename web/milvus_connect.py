import random

from pymilvus import connections, db, FieldSchema, DataType, CollectionSchema, Collection

MILVUS_HOST = "localhost"
MILVUS_PORT = '19530'

connection = connections.connect(user='minioadmin', password='minioadmin', host=MILVUS_HOST,
                                 port=MILVUS_PORT)


def milvus_connection():
    connection = connections.connect(alias='milvus_conn', user='minioadmin', password='minioadmin', host=MILVUS_HOST,
                                     port=MILVUS_PORT)
    return connection


# milvus function test
def milvus_func():
    # connection = milvus_connection()
    dbs = db.list_database(timeout=15)
    print(dbs)
    # database = db.create_database('book')
    db.using_database("book")
    # fields = [
    #     FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=False),
    #     FieldSchema(name="random", dtype=DataType.DOUBLE),
    #     FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=8)
    # ]
    # schema = CollectionSchema(fields, "hello_milvus is the simplest demo to introduce the APIs")
    # hello_milvus = Collection("hello_milvus", schema)
    #
    # data = add_data()
    #
    # hello_milvus.insert(data)
    # hello_milvus.flush()
    # ----- query ------
    hello_milvus = Collection("hello_milvus")
    # ---- 构建索引----
    index = {
        "index_type": "IVF_FLAT",
        "metric_type": "L2",
        "params": {"nlist": 128},
    }
    hello_milvus.create_index("embeddings", index)

    hello_milvus.load()
    result = hello_milvus.query(expr="random > -14", output_fields=["random", "embeddings"])
    print(result)


def add_data():
    data = [[i for i in range(5)],  # field pk
            [float(random.randrange(-20, -10)) for _ in range(5)],  # field random
            [[random.random() for _ in range(8)] for _ in range(5)]]
    print(data)
    return data


if __name__ == '__main__':
    milvus_func()
    # print(add_data())

