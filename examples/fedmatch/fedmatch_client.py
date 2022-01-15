"""
A federated semi-supervised learning client using FedMatch, and data samples on devices
are mostly unlabeled.

Reference:

Jeong et al., "Federated Semi-supervised learning with inter-client consistency and
disjoint learning", in the Proceedings of ICLR 2021.

https://arxiv.org/pdf/2006.12097.pdf
"""
from plato.clients import simple
from dataclasses import dataclass


@dataclass
class Report(simple.Report):
    """Report from a simple client, to be sent to the federated learning server."""
    client_id: int


class Client(simple.Client):
    """A fedmatch federated learning client who sends weight updates
    and the number of local epochs."""
    def __init__(self,
                 model=None,
                 datasource=None,
                 algorithm=None,
                 trainer=None):
        super().__init__(model, datasource, algorithm, trainer)
        self.helper_flag = False
        self.helpers = None
        self.sup_train = None
        self.unsup_train = None

    async def train(self):
        """ Fedmatch clients use different number of local epochs. """
        self.trainer.helpers = self.helpers

        report, weights = await super().train()
        return Report(report.num_samples, report.accuracy,
                      report.training_time, report.data_loading_time,
                      self.client_id), weights

    def load_payload(self, server_payload):
        """ Load model weights and helpers from server payload onto this client. """
        if self.helper_flag is False:
            self.algorithm.load_weights(server_payload)
            self.helper_flag = True
        else:
            self.algorithm.load_weights(server_payload[0])
            self.helpers = server_payload[1]
