from pymilvus import MilvusClient, Collection, connections

host = "localhost"
port = '19530'
milvus_client = MilvusClient(uri=f"http://{host}:{port}", user='', password='', timeout=15, db_name='text2vec')
connections.connect(host=host, port=port, user='', password='', timeout=15, db_name='text2vec')

collection_name = "text2vec_milvus"


def insert_data(data):
    """
    插入数据到集合中

    Parameters:
    - data: 要插入的数据，格式为 [{'id': id_value, 'embeddings': embeddings, 'keywords': keywords}, ...]
    """
    try:
        milvus_client.insert(collection_name, data)
        milvus_client.flush(collection_name)
        emb_collection = Collection(collection_name)
        emb_collection.load()
        print("Insertion successful.")
    except Exception as e:
        print(f"Failed to insert data: {e}")


def delete_data_by_expression(ids, filter_expr):
    print(f"Start to delete by expression {filter_expr} in collection {collection_name}")

    try:
        # Delete data based on expression
        milvus_client.delete(collection_name, pks=ids, filter=filter_expr)
        milvus_client.flush(collection_name)
        print("Deletion successful.")
    except Exception as e:
        print(f"Failed to delete data: {e}")


def query_data_by_expression(filter_expr, output_fields, limit=10):
    print(f"Start to query by expression {filter_expr} in collection {collection_name}")

    try:
        # Delete data based on expression
        query_result = milvus_client.query(collection_name, filter=filter_expr, output_fields=output_fields)

        query_result_limit = query_result[:limit]

        print("Query successful.")
    except Exception as e:
        print(f"Failed to Query data: {e}")

    return query_result_limit


def search_data_by_expression(encodes):
    # search_result = milvus_client.search(collection_name, expr)
    search_result = milvus_client.search({'collection_name': collection_name, 'data': encodes, 'output_fields': ['w_id', 'keywords']})
    return search_result


def delete_data_by_user_id(user_id='1000000'):
    filter_expression = f"user_id == {user_id}"
    limit = 2000
    output_fields = ["id"]
    res = query_data_by_expression(collection_name, filter_expression, output_fields, limit)

    res_array = [item['id'] for item in res]

    delete_data_by_expression(collection_name, pks=res_array, filter_expr='')


def delete_data_by_id(ids):
    delete_data_by_expression(collection_name, pks=ids, filter_expr='')


def generate_string_array(start, end):
    """
    生成指定范围内的字符串数组

    Parameters:
    - start: 范围起始值
    - end: 范围结束值（不包含）

    Returns:
    - 字符串数组
    """
    return [str(i) for i in range(start, end + 1)]


def test_query_data():
    filter_expression = "user_id == '1000000'"
    limit = 1
    output_fields = ["id", "user_id", "image_path"]

    res = query_data_by_expression(collection_name, filter_expression, output_fields, limit)

    print(res)


def test_delete_data():
    # 示例：生成范围字符串数组
    result_array = generate_string_array(1100122, 1100510)

    # 打印结果数组
    print(result_array)

    delete_data_by_id(result_array)


def test_insert_data():
    import random

    # 生成包含1024个随机整数的数组
    random_array = [random.randint(1, 1000) for _ in range(1024)]

    # 打印数组的前10个元素
    print(random_array[:10])

    # 示例数据，用于插入
    sample_data = [
        {'id': '1100122', 'user_id': '1000000', 'finger_print': 'sfasfdsfasdf', 'vector': random_array,
         'image_path': '/path/to/image1.jpg'},
        {'id': '1100123', 'user_id': '1000000', 'finger_print': 'sfasfdsfasdf1', 'vector': random_array,
         'image_path': '/path/to/image1.jpg'},
        # 添加更多数据项...
    ]

    # 调用插入方法
    insert_data(sample_data)


if __name__ == "__main__":
    test_insert_data()
