from model import Business2Vec


def train():
    b2v = Business2Vec()
    b2v.build()
    b2v.train()


if __name__ == '__main__':
    train()
