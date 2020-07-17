from cnn.network_input import NetworkInputSplit, NetworkInput
from cnn.network_history import NetworkHistory
import numpy as np
import os
from constants import Constants

def save_probs(filename, net_input: NetworkInput, probs, zone=False, truth=False):
    filename = os.path.join(Constants.log_dir, Constants.test_name, filename)
    fname = os.path.abspath(filename)
    dname = os.path.dirname(fname)
    if not os.path.isdir(dname) and not os.path.isfile(dname):
        os.makedirs(dname)

    ProxIds = net_input.ids[:, 0].reshape([len(net_input.ids), 1])
    fids = net_input.ids[:, 1].reshape([len(net_input.ids), 1])
    header = "ProxID, fid"
    format = "%s, %d"
    data = [ProxIds, fids]

    if zone:
        header += ", zone"
        format += ", %s"
        data.append(net_input.zones())

    if True:
        clinsigs = probs[..., 1].reshape([len(net_input.ids), 1])
        header += ", ClinSig"
        format += ", %.3f"
        data.append(clinsigs)

    if truth:
        ys = net_input.ys[..., 1].reshape([len(net_input.ids), 1])
        header += ", Truth"
        format += ", %d"
        data.append(ys)
    X = np.concatenate(data, axis=1)
    np.savetxt(fname=filename, X=X, delimiter=",", fmt=format, header=header)


def norm_to_aug(train_ni: NetworkInput, indexes, aug_ni: NetworkInput):
    aug_ids_pidfid = list(pid + '-' + str(fid) for pid, fid in aug_ni.ids)
    train_ids_pidfid = list(pid + '-' + str(fid) for pid, fid in train_ni.ids[indexes])
    return [i for i, id in enumerate(aug_ids_pidfid) if id in train_ids_pidfid]


# Boolean ys for StratifiedKFold
# net_input_ys_bool = net_input.ys[: , 1].astype(bool)

def bitlist_to_int(bitlist):
    out = 0
    for bit in bitlist:
        out = (out << 1) | bool(bit)
    return out

# Saves test / train / val set predictions
def dump_probs(test_name, history: NetworkHistory, train_val: NetworkInputSplit, test: NetworkInput, test_no, fold):
    best_auc = history.val_auc[history.best_epoch]

    for checkpoint in Constants.checkpoint_epochs:
        if checkpoint in history.test_probabilities:
            save_probs("test.epoch-%d.%s-%d-%d-0.%d.probabilities.csv" % (checkpoint, test_name, test_no, fold, int(best_auc * 100)),
                       test,
                       history.test_probabilities[checkpoint])

        if checkpoint in history.train_probabilities:
            save_probs("train.epoch-%d.%s-%d-%d.probabilities.csv" % (checkpoint, test_name, test_no, fold),
                       train_val.train_test,
                       history.train_probabilities[checkpoint],
                       zone=True,
                       truth=True)

        if checkpoint in history.val_probabilities:
            save_probs("val.epoch-%d.%s-%d-%d.probabilities.csv" % (checkpoint, test_name, test_no, fold),
                       train_val.val,
                       history.val_probabilities[checkpoint],
                       zone=True,
                       truth=True)

    save_probs("test.%s-%d-%d-0.%d.probabilities.csv" % (test_name, test_no, fold, int(best_auc * 100)),
               test,
               history.test_probabilities[history.best_epoch])

    save_probs("train.%s-%d-%d.probabilities.csv" % (test_name, test_no, fold),
               train_val.train_test,
               history.train_probabilities[history.best_epoch],
               zone=True,
               truth=True)

    save_probs("val.%s-%d-%d.probabilities.csv" % (test_name, test_no, fold),
               train_val.val,
               history.val_probabilities[history.best_epoch],
               zone=True,
               truth=True)