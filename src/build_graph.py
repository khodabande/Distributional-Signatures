import os
import sys
import random
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from itertools import product
from utils import loadWord2Vec, clean_str
from math import log
from sklearn import svm
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import cosine
from scipy.sparse import eye, vstack, bmat, diags
from main import parse_args, print_args, set_seed
from dataset.loader import _load_json, _meta_split, _get_reuters_classes


def load(args):
    train_classes, val_classes, test_classes = _get_reuters_classes(args)
    # load data
    all_data = _load_json(args.data_path)
    train_data, val_data, test_data = _meta_split(all_data, train_classes, val_classes, test_classes)
    all_data = train_data + val_data + test_data
    print('#train {}, #val {}, #test {}'.format(len(train_data), len(val_data), len(test_data)))
    
    ids = list(range(len(all_data)))
    names = None
    docs = [' '.join(x['text']) for x in all_data]
    labels = [x['label'] for x in all_data]
    train_size = len(train_data)
    val_size = len(val_data)
    test_size = len(test_data)
    return ids, names, docs, labels, train_size, val_size, test_size
    

def build_vocab(docs_words):
    word_id_map = {}
    vocab = []
    for words in docs_words:
        for w in words:
            if w not in word_id_map:
                word_id_map[w] = len(vocab)
                vocab.append(w)
    return vocab, word_id_map


def build_freq_matrix(docs_wids, word_id_map):
    rows = []
    cols = []
    data = []
    for i, doc in enumerate(docs_wids):
        rows.extend(i for _ in range(len(doc)))
        cols.extend(doc)
        data.extend(1 for _ in range(len(doc)))
    freq = sp.csr_matrix((data, (rows, cols)), shape=(len(docs_wids), len(word_id_map)))
    return freq


def get_tfidf(freq, idf):
    tfidf = freq.copy()
    rows, cols = tfidf.nonzero()
    tfidf.data = np.array(tfidf.data) * idf[cols]
    return tfidf


def get_label_ids(labels):
    label_id_map = {}
    label_list = []
    for label in labels:
        if label not in label_id_map:
            label_id_map[label] = len(label_list)
            label_list.append(label)
    return label_list, label_id_map


# def build_window_freq_matrix(docs_wids, vocab_size, window_size=20):
#     rows = []
#     cols = []
#     data = []
#     window_counter = 0
#     for i, wids in enumerate(docs_wids):
#         length = len(wids)
#         size = min(length, window_size)
#         for j in range(length - size + 1):
#             unique = list(set(wids[j:j+size])) # duplicates not included!
#             rows.extend(window_counter for _ in range(len(unique)))
#             cols.extend(wid for wid in unique)
#             data.extend(1 for _ in range(len(unique)))
#             window_counter += 1
#     wfm = sp.coo_matrix((data, (rows, cols)), shape=(window_counter, vocab_size))
#     wfm = wfm.transpose() * wfm
#     wfm = wfm.diagonal()
#     return wfm, window_counter


# def build_window_cofreq_matrix(docs_wids, vocab_size, window_size=20):
#     rows = []
#     cols = []
#     data = []
#     window_counter = 0
#     for i, wids in enumerate(docs_wids):
#         length = len(wids)
#         size = min(length, window_size)
#         for j in range(length - size + 1):
#             unique = wids[j:j+size] # duplicates included!
#             rows.extend(window_counter for _ in range(len(unique)))
#             cols.extend(wid for wid in unique)
#             data.extend(1 for _ in range(len(unique)))
#             window_counter += 1
#     wpm = sp.coo_matrix((data, (rows, cols)), shape=(window_counter, vocab_size))
#     wpm = wpm.transpose() * wpm
#     wpm = wpm - diags(wpm.diagonal())
#     return wpm, window_counter


def build_window_freqs(docs_wids, vocab_size, window_size=20):
    rows = []
    cols = []
    data = []
    window_counter = 0
    for i, wids in enumerate(docs_wids):
        length = len(wids)
        size = min(length, window_size)
        for j in range(length - size + 1):
            unique = list(set(wids[j:j+size])) # duplicates not included!
            rows.extend(window_counter for _ in range(len(unique)))
            cols.extend(wid for wid in unique)
            data.extend(1 for _ in range(len(unique)))
            window_counter += 1
    wpm = sp.coo_matrix((data, (rows, cols)), shape=(window_counter, vocab_size))
    wpm = wpm.transpose() * wpm
    wfm = wpm.diagonal()
    wpm = wpm - diags(wfm)
    return wfm, wpm, window_counter


def load_or_build_embedding(ds, vocab):
    # One-hot embedding
    # embd = eye(len(vocab))
    # return embd
    
    # Read Word Vectors
    # word_vector_file = 'data/glove.6B/glove.6B.300d.txt'
    # word_vector_file = 'data/corpus/' + dataset + '_word_vectors.txt'
    #_, embd, word_vector_map = loadWord2Vec(word_vector_file)
    # word_embeddings_dim = len(embd[0])
    try:
        word_vector_file = 'data/corpus/' + ds + '_word_vectors.txt'
        word_vec_vocab, embd, word_vec_id_map = loadWord2Vec(word_vector_file)
        word_embeddings_dim = len(embd[0])

        # word embedding matrix
        wm = np.matrix(embd)
        return word_vec_vocab, wm, word_vec_id_map
    except:
        print('Building embedding...')
        definitions = []
        for word in vocab:
            word = word.strip()
            synsets = wn.synsets(clean_str(word))
            word_defs = []
            for synset in synsets:
                syn_def = synset.definition()
                word_defs.append(syn_def)
            word_des = ' '.join(word_defs)
            if word_des == '':
                word_des = '<PAD>'
            definitions.append(word_des)

        tfidf_vec = TfidfVectorizer(max_features=1000)
        tfidf_matrix = tfidf_vec.fit_transform(definitions)
        tfidf_matrix_array = tfidf_matrix.toarray()

        word_vectors = []

        for i in range(len(vocab)):
            word = vocab[i]
            vector = tfidf_matrix_array[i]
            str_vector = []
            for j in range(len(vector)):
                str_vector.append(str(vector[j]))
            temp = ' '.join(str_vector)
            word_vector = word + ' ' + temp
            word_vectors.append(word_vector)

        string = '\n'.join(word_vectors)
        f = open('data/corpus/' + ds + '_word_vectors.txt', 'w')
        f.write(string)
        f.close()
        
        return load_or_build_embedding(ds, vocab)


def write_list(l, file):
    with open(file, 'w') as f:
        for item in l:
            f.write(str(item))
            f.write('\n')
    return True

def dump_obj(obj, file):
    with open(file, 'wb') as f:
        pkl.dump(obj, f)



if __name__ == '__main__':
    
    #if len(sys.argv) != 2:
    #    sys.exit("Use: python build_graph.py <dataset>")

    #datasets = ['20ng', 'R8', 'R52', 'ohsumed', 'mr']
    #dataset = sys.argv[1]

    #if dataset not in datasets:
    #    sys.exit("wrong dataset name")
    
    args = parse_args()

    print_args(args)

    set_seed(args.seed)
    
    dataset = 'reuters'

    print('Reading data...')
    ids, names, docs, labels, train_size, val_size, test_size = load(args)
    train_size += val_size

    #write_list(ids[:train_size], 'data/fs.' + dataset + '.train.index')
    #write_list(ids[train_size:], 'data/fs.' + dataset + '.test.index')
    #write_list(names, 'data/fs.' + dataset + '_shuffle.txt')
    write_list(docs, 'data/corpus/fs.' + dataset + '_shuffle.txt')
    write_list([train_size-val_size, val_size, test_size], 'data/fs.' + dataset + '.lens')
    print(train_size-val_size, val_size, test_size)

    print('Building vocab...')
    docs_words = [doc.split() for doc in docs]
    vocab, word_id_map = build_vocab(docs_words)
    docs_wids = [[word_id_map[w] for w in doc] for doc in docs_words]

    write_list(vocab, 'data/corpus/fs.' + dataset + '_vocab.txt')

    print('Frequencies...')
    freq_mat = build_freq_matrix(docs_wids, word_id_map)

    print('Embedding...')
    word_vec_vocab, word_mat, word_vec_id_map = load_or_build_embedding(dataset, vocab)
    filtered_words = sorted(list(set(word_vec_vocab).intersection(set(vocab))))
    filtered_words_id_map = {}
    for i, word in enumerate(filtered_words):
        filtered_words_id_map[word] = i
    print(len(filtered_words))
    filtered_word_vec_ids = [word_vec_id_map[word] for word in filtered_words]
    filtered_word_ids = [word_id_map[word] for word in filtered_words]
    freq_mat = freq_mat[:,filtered_word_ids]
    word_mat = word_mat[filtered_word_vec_ids,:]
    filtered_docs_wids = [[filtered_words_id_map[vocab[wid]] for wid in doc if vocab[wid] in filtered_words_id_map] for doc in docs_wids]
    print([len(doc) for doc in filtered_docs_wids][0:10])
            

    print('Label IDs...')
    label_list, label_id_map = get_label_ids(labels)
    label_ids = [label_id_map[l] for l in labels]
    label_mat = np.eye(len(label_list))

    write_list(label_list, 'data/corpus/fs.' + dataset + '_labels.txt')


    print('Feature vectors...')
    real_train_size = train_size - val_size

    #write_list(names[:real_train_size], 'data/fs.' + dataset + '.real_train.name')

    # train
    print('train')
    train_freq = freq_mat[:real_train_size]
    x = (train_freq / train_freq.sum(1)) * word_mat # for non one-hot embeddings
    # x = (train_freq / train_freq.sum(1)) # for one-hot embedding
    y = label_mat[label_ids[:real_train_size],:]

    # test
    print('test')
    test_freq = freq_mat[train_size:]
    tx = (test_freq / test_freq.sum(1)) * word_mat # for non one-hot embeddings
    # tx = (test_freq / test_freq.sum(1)) # for one-hot embedding
    ty = label_mat[label_ids[train_size:],:]

    # all (+words)
    print('all')
    train_freq = freq_mat[:train_size]
    allx = (train_freq / train_freq.sum(1)) * word_mat # for non one-hot embeddings
    # allx = (train_freq / train_freq.sum(1)) # for one-hot embedding
    ally = label_mat[label_ids[:train_size],:]
    #ally = label_mat[label_ids[:real_train_size],:]
    allx = vstack([allx, word_mat])
    ally = vstack([ally, sp.csr_matrix((len(vocab), len(label_list)))])

    #print(x.shape, y.shape, tx.shape, ty.shape, allx.shape, ally.shape)

    print('PMIs...')
    #window_freq, num_windows = build_window_freq_matrix(docs_wids, len(vocab))
    #window_cofreq, num_windows = build_window_cofreq_matrix(docs_wids, len(vocab))
    #window_freq, window_cofreq, num_windows = build_window_freqs(docs_wids, len(filtered_words))
    window_freq, window_cofreq, num_windows = build_window_freqs(filtered_docs_wids, len(filtered_words))

    # pmi as weights
    pmi = window_cofreq.copy()
    rows, cols = pmi.nonzero()
    pmi.data = np.clip(np.log(np.divide(pmi.data, window_freq[rows] * window_freq[cols] / float(num_windows))), 0, None)


    print('Adjacency matrix...')
    print(freq_mat.shape)
    app_mat = freq_mat.copy()
    print(app_mat.shape)
    app_mat[app_mat > 0] = 1
    print(app_mat.shape)
    word_freq_arr = np.asarray(app_mat.sum(0))[0]
    print(word_freq_arr.shape)
    idf_arr = np.log(float(len(docs)) / word_freq_arr)
    print(idf_arr.shape)
    tfidf_mat = get_tfidf(freq_mat, idf_arr)
    print(tfidf_mat.shape)

    node_size = train_size + len(vocab) + test_size

    adj = bmat([
        [None, tfidf_mat[:train_size], None],
        [tfidf_mat[:train_size].transpose(), pmi, tfidf_mat[train_size:].transpose()],
        [None, tfidf_mat[train_size:], None]
    ])

    # adj = bmat([
    #     [None, tfidf_mat[:train_size], None],
    #     [sp.csr_matrix((len(vocab), train_size)), pmi, sp.csr_matrix((len(vocab), test_size))],
    #     [None, tfidf_mat[train_size:], None]
    # ])


    dump_obj(x, "data/fs.ind." + dataset + ".x")
    dump_obj(y, "data/fs.ind." + dataset + ".y")
    dump_obj(tx, "data/fs.ind." + dataset + ".tx")
    dump_obj(ty, "data/fs.ind." + dataset + ".ty")
    dump_obj(allx, "data/fs.ind." + dataset + ".allx")
    dump_obj(ally, "data/fs.ind." + dataset + ".ally")
    dump_obj(adj, "data/fs.ind." + dataset + ".adj")



    # word vector cosine similarity as weights

    '''
    for i in range(vocab_size):
        for j in range(vocab_size):
            if vocab[i] in word_vector_map and vocab[j] in word_vector_map:
                vector_i = np.array(word_vector_map[vocab[i]])
                vector_j = np.array(word_vector_map[vocab[j]])
                similarity = 1.0 - cosine(vector_i, vector_j)
                if similarity > 0.9:
                    print(vocab[i], vocab[j], similarity)
                    row.append(train_size + i)
                    col.append(train_size + j)
                    weight.append(similarity)
    '''




# def main():
    

#     # initialize model
#     model = {}
#     model["ebd"] = ebd.get_embedding(vocab, args)
#     model["clf"] = clf.get_classifier(model["ebd"].ebd_dim, args)

#     if args.mode == "train":
#         # train model on train_data, early stopping based on val_data
#         train_utils.train(train_data, val_data, model, args)

#     elif args.mode == "finetune":
#         # sample an example from each class during training
#         way = args.way
#         query = args.query
#         shot = args.shot
#         args.query = 1
#         args.shot= 1
#         args.way = args.n_train_class
#         train_utils.train(train_data, val_data, model, args)
#         # restore the original N-way K-shot setting
#         args.shot = shot
#         args.query = query
#         args.way = way

#     # testing on validation data: only for not finetune
#     # In finetune, we combine all train and val classes and split it into train
#     # and validation examples.
#     if args.mode != "finetune":
#         val_acc, val_std = train_utils.test(val_data, model, args,
#                                             args.val_episodes)
#     else:
#         val_acc, val_std = 0, 0

#     test_acc, test_std = train_utils.test(test_data, model, args,
#                                           args.test_episodes)

#     if args.result_path:
#         directory = args.result_path[:args.result_path.rfind("/")]
#         if not os.path.exists(directory):
#             os.mkdirs(directory)

#         result = {
#             "test_acc": test_acc,
#             "test_std": test_std,
#             "val_acc": val_acc,
#             "val_std": val_std
#         }

#         for attr, value in sorted(args.__dict__.items()):
#             result[attr] = value

#         with open(args.result_path, "wb") as f:
#             pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)