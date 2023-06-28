"""
Check the model size of MobileNetV3 and the biggest and the smallest subnets of the MobileNetV3 search space.
Put this file under the ./examples/pfedrlnas/MobilenetV3 in the plato.
"""

import pickle
import sys

from transformer import Transformer

if __name__ == "__main__":
    model = Transformer(model_rate=1.0)
    payload = model.state_dict()
    payload_size = sys.getsizeof(pickle.dumps(payload)) / 1024**2
    print(payload_size)
