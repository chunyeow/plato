"""
A FedAvg session implemention client dropping according to the hard constraints.
"""
import pickle
import sys
import logging
import copy

import numpy as np
import ptflops


from plato.servers import fedavg
from plato.config import Config


class Server(fedavg.Server):
    """
    A customized server to simulate the constraints vairance during federated learning.
    """

    def __init__(
        self,
        model=None,
        datasource=None,
        algorithm=None,
        trainer=None,
    ):
        # pylint:disable=too-many-arguments
        super().__init__(model, datasource, algorithm, trainer)
        self.limitation = np.zeros(
            (Config().clients.total_clients, 2)
        )
        self.participated_clients = 0
#        if (
#            hasattr(Config().parameters.limitation, "activated")
#            and Config().parameters.limitation.activated
#        ):
#            limitation = Config().parameters.limitation
#            self.limitation[:,0] = np.random.uniform(
#                limitation.min_size,
#                limitation.max_size,
#                (Config().clients.total_clients),
#            )
#            self.limitation[:,1] = np.random.uniform(
#                limitation.min_flops,
#                limitation.max_flops,
#                (Config().clients.total_clients),
#            )

    def generate_customized_model(self):
        """
        Do a simulation to generate the customized model, for calculating flops.
        """
        return self.algorithm.model

    def choose_clients(self, clients_pool, clients_count):
        selected_clients = super().choose_clients(clients_pool, clients_count)

#        if not (
#            hasattr(Config().parameters.limitation, "activated")
#            and Config().parameters.limitation.activated
#        ):
        return selected_clients
#        selected_clients_return = []
#        for client_id in selected_clients:
#            server_response = {
#                "id": client_id,
#                "current_round": self.current_round,
#            }
#            server_response = self.customize_server_response(
#                server_response, client_id=self.selected_client_id
#            )
#
#            payload = self.algorithm.extract_weights()
#            payload = self.customize_server_payload(payload)
#            size = sys.getsizeof(pickle.dumps(payload)) / 1024**2
#            model = copy.deepcopy(self.generate_customized_model())
#            macs, _ = ptflops.get_model_complexity_info(
#                model,
#                (3, 32, 32),
#                as_strings=False,
#                print_per_layer_stat=False,
#                verbose=False,
#            )
#            macs /= 1024**2
#            if (size <= self.limitation[client_id - 1, 0]) and (
#                macs <= self.limitation[client_id - 1, 1]
#            ):
#                selected_clients_return.append(client_id)
#        self.participated_clients = len(selected_clients_return)
#        logging.info("[%s] Re Selected clients: %s", self, selected_clients_return)
#        self.clients_per_round = len(selected_clients_return)
#        if len(selected_clients_return)==0:
#            selected_clients_return=self.choose_clients(clients_pool,clients_count)
#
#        return selected_clients_return

    def get_logged_items(self) -> dict:
        logged_items = super().get_logged_items()
        logged_items["chosen"] = self.participated_clients
        return logged_items

    async def wrap_up(self) -> None:
        await super().wrap_up()
        self.clients_per_round = Config().clients.per_round
