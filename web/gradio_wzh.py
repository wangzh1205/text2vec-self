import gradio as gr
from text2vec import SentenceModel
import multimodal_search_dao as dao
import hashlib
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
    print("type: {}, code:{}".format(type(code), code))
    codes = code.tolist()
    print('----------------numpy.ndarray 转换为 list-------------------')
    print("type: {}, code:{}".format(type(codes), codes))
    return code.tolist()


def vector(word, is_show):
    codes = encode(word)
    wid = hashlib.md5(word.encode(encoding='UTF-8')).hexdigest()
    insert_milvus(wid, codes, word)
    if is_show:
        return wid, word, codes
    else:
        return wid, None, codes


def query_milvus(word, limitNum):
    codes = encode(word)
    print("type: {}, codes:{}".format(type(codes), codes))
    # expr = ['vector': codes]
    # result = dao.search_data_by_expression(codes)
    result = milvus_coll.search_data(codes, limitNum)
    print(result)
    return paras_search_result(result)
    # return result


def paras_search_result(hit_result):
    print(len(hit_result))
    hits = hit_result[0]
    for hit in hits:
        # print("type={}, data={}".format(type(hit), hit))
        print("hit id={}, distance={}, w_id={}, keyword={}".format(hit.id, hit.distance, hit.w_id, hit.keywords))
    # results = [
    #     {
    #         "id": hit.id,
    #         "distance": hit.distance,
    #         "wid": hit.w_id,
    #         "word": hit.keywords
    #     }
    #     for hit in hits
    # ]
    results = [
        [hit.id, hit.distance, hit.w_id, hit.keywords]
        for hit in hits
    ]
    return results


def main():
    input_word = gr.Text(label='输入待向量化文字')
    input_show = gr.Checkbox(label='是否回显', value=True)

    output_pk = gr.Text(label='主键PK值')
    output_embeddings = gr.Text(label='向量化结果')
    output_word = gr.Text(label='原始数据')
    app = gr.Interface(
        fn=vector,
        inputs=[input_word, input_show],
        outputs=[output_pk, output_embeddings, output_word],
        title='珍林-向量化demo')
    app.launch()
    # app.launch(share=True)


def main1():
    with gr.Blocks() as app:
        gr.Markdown('select input or select search data')
        with gr.Tab("generate input data"):
            with gr.Column():
                input_word = gr.Text(label='输入待向量化文字')
                input_show = gr.Checkbox(label='是否回显', value=True)
            with gr.Row():
                output_pk = gr.Text(label='主键PK值')
                output_word = gr.Text(label='原始数据')
                output_embeddings = gr.Text(label='向量化结果')

            input_button = gr.Button("生成向量化")
            input_button.click(fn=vector, inputs=[input_word, input_show], outputs=[output_pk, output_word,
                                                                                    output_embeddings])

        with gr.Tab("search data"):
            input_word = gr.Text(label="输入要查询的数据")
            input_slider = gr.Slider(label="选择要显示的条数", value=3, minimum=1, maximum=10, step=1)
            with gr.Row():
                # output_hit = gr.Text(label="匹配结果")
                output_hit = gr.DataFrame(headers=["id", "distance", "wid", "word"])

            search_button = gr.Button("搜索")
            search_button.click(fn=query_milvus, inputs=[input_word, input_slider], outputs=output_hit)
    app.launch(share=True)


if __name__ == '__main__':
    # main()
    # vector('我要用花呗')
    main1()
