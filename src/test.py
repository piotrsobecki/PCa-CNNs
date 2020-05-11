import sys

def main(argv):
    from constants import Constants
    from test_utils import bitlist_to_int, dump_probs, norm_to_aug
    import numpy as np
    from cnn.network_input import NetworkInputSplit
    from cnn.network_input_provider import NetworkInputProvider
    from cnn.threaded_batch_iter import _gen_batches
    from cnn.network_factory import NetworkFactory
    from cnn.network_trainer import NetworkTrainer
    from sklearn.model_selection import StratifiedKFold

    print("PACKAGES LOADED")
    Constants()
    sys.path.append(Constants.rel)
    sys.path.append(Constants.rel + 'lib')

    cv_seed = 1

    net_input_provider = NetworkInputProvider(Constants.dataset_dir, Constants.test_name)
    net_input_provider_test = NetworkInputProvider(Constants.dataset_test_dir, Constants.test_name)

    net_input = net_input_provider.get_network_input("train", do_preprocessing=False)
    augmented_input = net_input_provider.get_network_input("augmented", do_preprocessing=False)
    net_input_test = net_input_provider_test.get_network_input("test", do_preprocessing=False)

    print(net_input.xs.keys())
    # print(augmented_input.xs.keys())
    print(net_input_test.xs.keys())

    net_input.setup_shapes()
    augmented_input.setup_shapes()
    net_input_test.setup_shapes()


    lesions_types = [str(bitlist_to_int(net_input.xs['location'][i])) + '-' + str(int(clinsig[1])) for i, clinsig in
                     enumerate(net_input.ys)]
    # lesions_types = net_input.ys[...,1]
    # Build network
    net_factory = NetworkFactory(Constants.architecture)

    network = net_factory.build_network([*_gen_batches(net_input, 1, False)][0].input_shape)

    folds = Constants.cv_folds
    tests = 50
    kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=cv_seed)
    fold = 0
    aucs = np.zeros(folds)

    for test_no in range(tests):

        # Stratified CV
        for train_index, val_index in kf.split(net_input.ids, lesions_types):
            print('CV [%02d]' % (fold))
            train = augmented_input.from_ids(norm_to_aug(net_input, train_index, augmented_input))
            train_test = net_input.from_ids(train_index)
            val = net_input.from_ids(val_index)

            network.clear_graph()
            net_input_split = NetworkInputSplit(train=train, train_test=train_test, val=val)
            net_trainer = NetworkTrainer(network, net_input_split)
            history = net_trainer.train(net_input_test)

            best_auc = history.val_auc[history.best_epoch]

            print('BEST EPOCH VAL AUC: %.5f' % best_auc)
            print(history.row(history.best_epoch))
            history.dump('HISTORY-%s/epochs-%d-%d.log' % (Constants.test_name, test_no, fold))
            dump_probs(Constants.test_name, history, net_input_split, net_input_test, test_no, fold)

            aucs[fold] = history.val_auc[history.best_epoch]
            fold += 1
        fold = 0
        print('TEST NO: %.1f' % test_no)
        f = open("cnn-m5.txt", "a+")
        f.write('{:.3f}\n'.format(np.mean(aucs)))
        f.close()


if __name__ == '__main__':
    main(sys.argv)
