"""
Check the model size of MobileNetV3 and the biggest and the smallest subnets of the MobileNetV3 search space.
Put this file under the ./examples/pfedrlnas/MobilenetV3 in the plato.
"""

import torchvision
import pickle
import sys

# import fedtools
# from model.mobilenetv3_supernet import NasDynamicModel
from ptflops import get_model_complexity_info
import mobilenetv3

if __name__ == "__main__":

    model = mobilenetv3.MobileNetV3(mode="large", input_size=32, classes_num=10)
    macs, params = get_model_complexity_info(
        model, (3, 32, 32), as_strings=False, print_per_layer_stat=False, verbose=False
    )
    payload = model.state_dict()
    payload_size = sys.getsizeof(pickle.dumps(payload)) / 1024**2
    print(payload_size, macs)
    # model = mobilenetv3.MobileNetV3(mode="small", input_size=32, classes_num=10)
    # macs, params = get_model_complexity_info(
    #     model, (3, 32, 32), as_strings=True, print_per_layer_stat=True, verbose=True
    # )
    # payload = model.state_dict()
    # payload_size = sys.getsizeof(pickle.dumps(payload)) / 1024**2
    # print(payload_size, macs)
    # model = NasDynamicModel()
    # subnet_config = model.sample_max_subnet()
    # model.set_active_subnet(
    #     subnet_config["resolution"],
    #     subnet_config["width"],
    #     subnet_config["depth"],
    #     subnet_config["kernel_size"],
    #     subnet_config["expand_ratio"],
    # )
    # flops = model.compute_active_subnet_flops()
    # subnet = fedtools.sample_subnet_w_config(model, subnet_config, True)
    # payload = subnet.state_dict()
    # payload_size = sys.getsizeof(pickle.dumps(payload)) / 1024**2
    # print(payload_size, flops)
    # subnet_config = model.sample_min_subnet()
    # model.set_active_subnet(
    #     subnet_config["resolution"],
    #     subnet_config["width"],
    #     subnet_config["depth"],
    #     subnet_config["kernel_size"],
    #     subnet_config["expand_ratio"],
    # )
    # flops = model.compute_active_subnet_flops()
    # subnet = fedtools.sample_subnet_w_config(model, subnet_config, True)
    # payload = subnet.state_dict()
    # payload_size = sys.getsizeof(pickle.dumps(payload)) / 1024**2
    # print(payload_size, flops)
