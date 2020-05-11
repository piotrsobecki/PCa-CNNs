from cnn.architectures.CNN_VGG_SIMPLE import build_vgg_simple
from cnn.architectures.CNN_VGG_MODALITIES import build_vgg_modalities
from cnn.architectures.CNN_VGG_PIRADS import build_vgg_pirads
from constants import NetStructure
from cnn.network import Network

class NetworkFactory:
    net_structure = NetStructure

    def __init__(self, net_structure: NetStructure):
        self.net_structure = net_structure

    def build_network(self, net_input_shape) -> Network:
        network = None
        if self.net_structure == NetStructure.CNN_VGG_SIMPLE:
            network = build_vgg_simple(net_input_shape)
        elif self.net_structure == NetStructure.CNN_VGG_MODALITIES:
            network = build_vgg_modalities(net_input_shape)
        elif self.net_structure == NetStructure.CNN_VGG_PIRADS:
            network = build_vgg_pirads(net_input_shape)
        return network
