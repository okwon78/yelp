from model import Business2Vec


def train():
    b2v = Business2Vec()
    b2v.build(vector_dim=10, learn_rate=0.1)
    b2v.train()


if __name__ == '__main__':
    train()
