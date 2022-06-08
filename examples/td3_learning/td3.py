"""
A federated learning training session with clients running td3
"""
import logging

import td3_learning_client
import td3_learning_trainer
import td3_learning_server
import td3_learning_model
import td3_learning_algorithm
import globals

from torch import nn

#to run
#python examples/td3_learning/td3.py -c examples/td3_learning/td3_FashionMNIST_lenet5.yml


#evaluations = [td3_learning_trainer.Trainer.evaluate_policy(trainer)]
evaluations = []

def main():
    """ A Plato federated learning training session with clients running TD3. """
    logging.info("Starting RL Environment's process.")

    """ A Plato federated learning training session using a custom model. """
    # TODO: there should be an actor and critic model communicated with clients.
    # TODO: Not only one single model. 
    # TODO: This model should be two models. How? I still do not know.
    #model = nn.Sequential(
       # nn.Linear(28, 128),
        #nn.ReLU(),
        #nn.Linear(128, 128),
        #nn.ReLU(),
        #nn.Linear(128, 10),
    #)

    model = td3_learning_model.Model(globals.state_dim, globals.action_dim, globals.max_action)
    trainer = td3_learning_trainer.Trainer(globals.state_dim, globals.action_dim, globals.max_action, model)
    algorithm = td3_learning_algorithm.Algorithm
    client = td3_learning_client.RLClient(model = model, trainer=trainer, algorithm = algorithm)
    server = td3_learning_server.TD3Server(model=model, algorithm = algorithm)
    server.run(client)

if __name__ == "__main__":
    main()
