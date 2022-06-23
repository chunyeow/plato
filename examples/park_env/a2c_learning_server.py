"""
A federated learning server using federated averaging to train Actor-Critic models.

To run this example:

python examples/customized/custom_server.py -c examples/customized/server.yml
"""

import logging
import asyncio
from plato.servers import fedavg
from plato.config import Config
import pickle
import random
import numpy as np


class A2CServer(fedavg.Server):
    """ Federated learning server using federated averaging to train Actor-Critic models. """
    """ A custom federated learning server. """

    def __init__(self, algorithm_name, env_name, model = None, trainer = None, algorithm = None):
        super().__init__(trainer = trainer, algorithm = algorithm, model = model)
        self.algorithm_name = algorithm_name
        self.env_name = env_name
        #self.updates = None
        logging.info("A custom server has been initialized.")
        
    async def federated_averaging(self, updates):
        """Aggregate weight updates from the clients using federated averaging."""
        print("WE ARE FEDERATED AVERAGING!!!")
        print("WE ARE FEDERATED AVERAGING!!!")

        weights_received = self.compute_weight_deltas(updates)

        #actor_loss_list = [report.actor_loss for (_, report, _, _) in updates]

        samples_list = []
        actor_loss_list = []
        critic_loss_list = []

        for (_, report, _, _) in updates:
            samples_list.append(report.num_samples)
            actor_loss_list.append(report.actor_loss)
            critic_loss_list.append(report.critic_loss)

        self.total_samples = sum(samples_list)
        percentile = Config().server.percentile

        #get the threshold for aggregation
        critic_percentile_aggregation = np.percentile(np.array(critic_loss_list), percentile)
        actor_percentile_aggregation = np.percentile(np.array(actor_loss_list), percentile)

        print("ACTOR LOSS LIST IS THIS")
        print(actor_loss_list)
        print("CRITIC LOSS LIST IS THIS")
        print(critic_loss_list)

        print("ACTOR PERCENTILE IS THIS")
        print(actor_percentile_aggregation)
        print("CRITIC PERCENTILE IS THIS")
        print(critic_percentile_aggregation)

      
        # Perform weighted averaging for both Actor and Critic
        actor_avg_update = {
            name: self.trainer.zeros(weights.shape)
            for name, weights in weights_received[0][0].items()
        }
        critic_avg_update = {
            name: self.trainer.zeros(weights.shape)
            for name, weights in weights_received[0][1].items()
        }

        for i, update in enumerate(weights_received):
            __, report, __, __ = updates[i]
            actor_loss = report.actor_loss
            critic_loss = report.critic_loss

            update_from_actor, update_from_critic = update

            if actor_loss > actor_percentile_aggregation:
                print("ACTOR LOSS IS THIS %f SO WE ARE HERE" %actor_loss)
                for name, delta in update_from_actor.items():
                    actor_avg_update[name] += delta 

            if critic_loss > critic_percentile_aggregation:
                print("CRITIC LOSS IS THIS %f so we are here" %critic_loss)
                for name, delta in update_from_critic.items():
                    critic_avg_update[name] += delta 
            # Yield to other tasks in the server
            await asyncio.sleep(0)
        
        return actor_avg_update, critic_avg_update

    def save_to_checkpoint(self):
        """ Save a checkpoint for resuming the training session. """
        checkpoint_path = Config.params['checkpoint_path']

        copy_algorithm = self.algorithm_name
        if '_' in copy_algorithm:
            copy_algorithm= copy_algorithm.replace('_', '')
        
        env_algorithm = self.env_name+copy_algorithm
        model_name = env_algorithm
        if '_' in model_name:
            model_name.replace('_', '')
        filename = f"checkpoint_{model_name}_{self.current_round}.pth"
        logging.info("[%s] Saving the checkpoint to %s/%s.", self,
                     checkpoint_path, filename)
        self.trainer.save_model(filename, checkpoint_path)
        self.save_random_states(self.current_round, checkpoint_path)

        # Saving the current round in the server for resuming its session later on
        with open(f"{checkpoint_path}/current_round.pkl",
                  'wb') as checkpoint_file:
            pickle.dump(self.current_round, checkpoint_file)

   # async def process_clients(self, client_info):
    #    print("WE WILL NOW PROCESS AND SELECT OUR OWN CLIENTS!!!!!!!!!!!!!!!!!")
        #TODO
        #Client_info holds a tuple of (useless, reports, useless, useless)
        #Interested in the second element of tuple

    """def choose_clients(self, clients_pool, clients_count):
       
        assert clients_count <= len(clients_pool)

        print("IN CHOOSE CLIENTS!!!!")
        print(self.updates) #notice how this is none since no report is returned at this point
        #TODO
        If we want to aggregate CERTAIN clients out of ALL clients 
        based on their actor and loss rate, we need to
        receive a report from the ALL clients FIRST and then select them
        
        selected_clients = []

        if Config().server.random:
            random.setstate(self.prng_state)
            
            # Select clients randomly
            selected_clients = random.sample(clients_pool, clients_count)

            self.prng_state = random.getstate()
        else:
            print("Custom selection started!")
            if(self.current_round <= 8):
                selected_clients.append(clients_pool[0])
                selected_clients.append(clients_pool[1])
            else:
                random.setstate(self.prng_state)
                
                selected_clients = random.sample(clients_pool, clients_count)

                self.prng_state = random.getstate()

        logging.info("[%s] Selected clients: %s", self, selected_clients)
        return selected_clients 
    """
    