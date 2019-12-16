from model import Business2Vec


def train():
    b2v = Business2Vec()
    b2v.build()
    b2v.train()
    b2v.retrieval_embeddings()
    b2v.calc_knn_annoy()


if __name__ == '__main__':
    train()
