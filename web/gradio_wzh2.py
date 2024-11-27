import gradio as gr
from text2vec import SentenceModel
import multimodal_search_dao as dao
import milvus_create_collection as milvus_coll

m = SentenceModel()


def insert_milvus(*data):
    print("type: {}, code:{}".format(type(data), data))
    if data:
        pk = data[0]
        em = data[1]
        word = data[2]
        dao.insert_data([{'w_id': pk, 'embeddings': em, 'keywords': word}])


def encode(word):
    code = m.encode(word)
    codes = code.tolist()
    return codes


def query_milvus(word):
    codes = encode(word)
    print("type: {}, codes:{}".format(type(codes), codes))
    # expr = ['vector': codes]
    # result = dao.search_data_by_expression(codes)
    result = milvus_coll.search_data(codes)
    print(result)


def main():
    input_word = gr.Text(label='输入待向量化文字')
    input_show = gr.Checkbox(label='是否回显', value=True)

    output_pk = gr.Text(label='主键PK值')
    output_embeddings = gr.Text(label='向量化结果')
    output_word = gr.Text(label='原始数据')
    app = gr.Interface(
        fn=encode,
        inputs=[input_word, input_show],
        outputs=[output_pk, output_embeddings, output_word],
        title='珍林-向量化demo')
    app.launch()
    # app.launch(share=True)


if __name__ == '__main__':
    # main()
    query_milvus('吃瓜')
