

import sys

sys.path.append('..')
from text2vec import SentenceModel


def main():
    # Create a large list of sentences
    sentences = ["This is sentence {}".format(i) for i in range(10000)]
    model = SentenceModel("shibing624/text2vec-base-chinese")
    print(f"Sentences size: {len(sentences)}, model: {model}")

    # Start the multi processes pool on all available CUDA devices
    pool = model.start_multi_process_pool()

    # Compute the embeddings using the multi processes pool
    emb = model.encode_multi_process(sentences, pool, normalize_embeddings=True)
    print(f"Embeddings computed. Shape: {emb.shape}")

    # Optional: Stop the process in the pool
    model.stop_multi_process_pool(pool)


if __name__ == "__main__":
    main()
