import wget
import zipfile
import numpy as np
import scipy.sparse as sp

def download(url, savepath):
    wget.download(url, str(savepath))


def unzip(zippath, savepath):
    zip = zipfile.ZipFile(zippath)
    zip.extractall(savepath)
    zip.close()


def map_data(data):
    u = list(set(data))
    id_dict = {old: new for new, old in enumerate(sorted(u))}
    data = np.array([id_dict[x] for x in data])
    return data, id_dict, len(u)


def trainvaltest_split(ratings, u_map, v_map, num_users, num_items):
    num_test = int(np.ceil(ratings.shape[0]) * 0.1)
    num_val = int(np.ceil(ratings.shape[0]) * 0.9 * 0.05)
    num_train = int(ratings.shape[0] - num_val - num_test)

    uvr_triplet = np.array([[u, v, r] for u, v, r in zip (u_map, v_map, ratings)])
    uv_index = np.array([u*num_items + v for u, v, _ in uvr_triplet])

    # index of 1D matrix
    train_idx = uv_index[:num_train].astype(np.int32)
    val_idx = uv_index[num_train:num_train+num_val].astype(np.int32)
    test_idx = uv_index[num_train+num_val:].astype(np.int32)

    # train/val/test_data = 0: u_idx, 1: v_idx, 2: labels
    train_data = uvr_triplet[:num_train].transpose()
    val_data = uvr_triplet[num_train:num_train+num_val].transpose()
    test_data = uvr_triplet[num_train+num_val:].transpose()

    # Create labels
    labels = create_1Dlabels(ratings, num_users, num_items, u_map, v_map)
    train_data[2] = labels[train_idx]
    val_data[2] = labels[val_idx]
    test_data[2] = labels[test_idx]

    train_label_data = labels[train_idx].astype(np.float32) + 1.
    val_label_data = labels[val_idx].astype(np.float32) + 1.
    test_label_data = labels[test_idx].astype(np.float32) + 1.

    train_rating_matrix = sp.csr_matrix((train_label_data, [train_data[0], train_data[1]]),
                                        shape=[num_users, num_items], dtype=np.float32)

    val_rating_matrix = sp.csr_matrix((val_label_data, [val_data[0], val_data[1]]),
                                        shape=[num_users, num_items], dtype=np.float32)

    test_rating_matrix = sp.csr_matrix((test_label_data, [test_data[0], test_data[1]]),
                                        shape=[num_users, num_items], dtype=np.float32)

    return train_rating_matrix, val_rating_matrix, test_rating_matrix


def create_1Dlabels(ratings, num_users, num_items, u_map, v_map):
    # No rating == rating: -1. Rating: 0~4
    labels = np.full((num_users, num_items), -1, dtype=np.int32)
    labels[u_map, v_map] = ratings - 1

    return labels.reshape(-1)
