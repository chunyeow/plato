"""
The implementation for the BYOL [1] method.

The official code: https://github.com/lucidrains/byol-pytorch

The third-party code: https://github.com/sthalles/PyTorch-BYOL


Reference:

[1]. https://arxiv.org/pdf/2006.07733.pdf

Source code: https://github.com/lucidrains/byol-pytorch
"""

import byol_net
import byol_trainer
import byol_server
import fedavg_byol

from plato.clients import ssl_simple as ssl_client


def main():
    """ A Plato federated learning training session using the SimCLR algorithm.
        This implementation of simclr utilizes the general setting, i.e.,
        removing the final fully-connected layers of model defined by
        the 'model_name' in config file.
    """
    trainer = byol_trainer.Trainer
    algorithm = fedavg_byol.Algorithm
    byol_model = byol_net.BYOL()
    client = ssl_client.Client(model=byol_model,
                               trainer=trainer,
                               algorithm=algorithm)
    server = byol_server.Server(model=byol_model,
                                trainer=trainer,
                                algorithm=algorithm)

    server.run(client)


if __name__ == "__main__":
    main()