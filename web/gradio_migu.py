import gradio as gr
from text2vec import SentenceModel
from pymilvus import connections, db, FieldSchema, DataType, CollectionSchema, Collection
import milvus_create_collection_migu_bge as migu_bge


m = SentenceModel()

MILVUS_HOST = "localhost"
MILVUS_PORT = '19530'
DATA_BASE = 'migu_text'
MIGU_COLLECTION_NAME = ('migu_text_collection')
MIGU_BGE_COLLECTION_NAME = ('migu_bge_text_collection')
connection = connections.connect(user='minioadmin', password='minioadmin', host=MILVUS_HOST,
                                 port=MILVUS_PORT)


def encode(word):
    code = m.encode(word)
    return code.tolist()


def query_milvus(word, radio, limitNum, score):
    if radio == 'text2vec':
        embeddings = encode(word)
        result = search_data(embeddings, limitNum, score, MIGU_COLLECTION_NAME)
    else:
        embeddings = migu_bge.migu_embedding_req(word)
        if embeddings is not None:
            result = search_data(embeddings, limitNum, score, MIGU_BGE_COLLECTION_NAME)
    return paras_search_result(result)


def search_data(embeddings, limit, score, collection_name):
    db.using_database(DATA_BASE)
    collection = Collection(collection_name)
    collection.load()
    query_item = {
        "data": [embeddings],
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


def paras_search_result(hit_result):
    hits = hit_result
    for hit in hits:
        print("hit id={}, distance={}, question_type={}, question_text={}, answer={} "
              .format(hit.id, hit.distance, hit.question_type, hit.question_text, hit.answer))
    results = [
        [hit.id, hit.distance, hit.question_type, hit.question_text, hit.answer]
        for hit in hits
    ]
    return results


def main():
    with gr.Blocks() as app:
        input_word = gr.Text(label="输入要匹配的问题")
        input_radio = gr.Radio(['text2vec', 'migu-bge'], label='选择向量模型', value='text2vec')
        input_num = gr.Slider(label="要显示的条数", value=3, minimum=1, maximum=10, step=1)
        input_score = gr.Slider(label="得分值", value=0.50, minimum=0, maximum=1, step=0.0001)
        with gr.Row():
            # output_hit = gr.Text(label="匹配结果")
            output_hit = gr.DataFrame(headers=["id", "distance", "问题类型", "问题描述", "问题答案"])

        search_button = gr.Button("搜索")
        search_button.click(fn=query_milvus, inputs=[input_word, input_radio, input_num, input_score], outputs=output_hit)

    # app.launch()
    app.launch(share=True)


if __name__ == '__main__':
    main()
    # vector('我要用花呗')
    # main1()
