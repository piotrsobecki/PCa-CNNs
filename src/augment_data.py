import sys
from cnn.network_input_provider import NetworkInputProvider
from constants import Constants

print("PACKAGES LOADED")

Constants()
sys.path.append(Constants.rel)
sys.path.append(Constants.rel + 'lib')

# Provide network input by preprocessing metadata and images
net_input_provider = NetworkInputProvider(Constants.dataset_dir, Constants.test_name)
net_input_provider_test = NetworkInputProvider(Constants.dataset_test_dir, Constants.test_name)

net_input_provider.get_network_input("train", discard_corrupted=False,  do_preprocessing=True)
net_input_provider.get_network_input("augmented", discard_corrupted=False,  do_preprocessing=True, do_augment=True)
net_input_provider_test.get_network_input("test", discard_corrupted=False,  do_preprocessing=True)

