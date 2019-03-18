from vecs_io import *
from transformer import *
from hash import RandomProjection
from sorter import *


def chunk_compress(pq, vecs):
    chunk_size = 1000000
    compressed_vecs = np.empty(shape=(vecs.shape[0], pq.L), dtype=np.float32)
    for i in tqdm.tqdm(range(math.ceil(len(vecs) / chunk_size))):
        compressed_vecs[i * chunk_size: (i + 1) * chunk_size, :] = pq.compress(
            vecs[i * chunk_size: (i + 1) * chunk_size, :].astype(dtype=np.float32))
    return compressed_vecs


def execute(pq, X, T, Q, G, metric, train_size=100000):
    np.random.seed(123)
    print("# Ranking metric {}".format(metric))
    print("# " + pq.class_message())
    if T is None:
        pq.fit(X[:train_size].astype(dtype=np.float32))
    else:
        pq.fit(T.astype(dtype=np.float32))

    print('# Compressing items...')
    vecs_encoded = chunk_compress(pq, X)
    query_encoded = chunk_compress(pq, Q)
    print(vecs_encoded.shape)
    print("# Sorting items...")
    Ts = [2 ** i for i in range(2 + int(math.log2(len(X))))]
    recalls = BatchSorter(vecs_encoded, query_encoded, X, G, Ts,
                          metric=metric, batch_size=200).recall()
    print("# Searching...\n")

    # table = PrettyTable()
    # table.field_names = ["Expected Items", "Overall time",
    #                      "AVG Recall", "AVG precision", "AVG error", "AVG items"]
    # for i, (t, recall) in enumerate(zip(Ts, recalls)):
    #     table.add_row([2 ** i, 0, recall, recall * len(G[0]) / t, 0, t])

    # print(table)
    print("{:^15}{:^15}{:^30}{:^30}{:^10}{:^15}".format(
        "expected items", "overall time", "avg recall", "avg precision", "avg error", "avg items"))
    for i, (t, recall) in enumerate(zip(Ts, recalls)):
        print("{:^15}{:^15}{:^30}{:^30}{:^10}{:^15}".format(
            2**i, 0, recall, recall * len(G[0]) / t, 0, t))


def parse_args():
    # override default parameters with command line parameters
    import argparse
    parser = argparse.ArgumentParser(
        description='Process input method and parameters.')
    parser.add_argument('--codelength', type=int, default=16,
                        help='choose the number of PQ tables')
    parser.add_argument('--dataset', type=str,
                        default="netflix", help='choose data set name')
    parser.add_argument('--topk', type=int, default=20,
                        help='required topk of ground truth')
    parser.add_argument('--metric', type=str, default="product",
                        help='metric of ground truth')
    args = parser.parse_args()
    return args.dataset, args.topk, args.codelength, args.metric


if __name__ == '__main__':

    top_k = 20
    dataset = 'netflix'
    metric = "product"
    folder = 'data/'
    code_length = 128

    import sys
    if len(sys.argv) > 3:
        dataset, top_k, code_length, metric = parse_args()
    else:
        import warnings
        warnings.warn("Using default Parameters ")
    print("# Parameters: #dataset = {}, topK = {}, code_length = {}, metric = {}"
          .format(dataset, top_k, code_length, metric))

    def raw():
        X, T, Q, G = loader(dataset, top_k, metric, folder)
        X, Q = scale(X, Q)
        pq = RandomProjection(code_length)
        execute(pq, X, T, Q, G, metric)

    raw()
