"""
A federated learning client for td3.
"""
import logging
from dataclasses import dataclass
from plato.clients import simple
import numpy as np
import pybullet_envs

@dataclass
class Report(simple.Report):
    """A client report to be sent to the federated learning server."""
    client_id: int
    #average_reward: int


class RLClient(simple.Client):
    
    def __init__(self, trainer = None, model = None, algorithm = None):
        #print("are we even ever in this consturctor line 30 of td_client")
        super().__init__(model=model, algorithm=algorithm, trainer=trainer)
        logging.info("A custom client has been initialized!")

    async def train(self):
         
        report, weights = await super().train()
      
        return Report(report.num_samples, report.accuracy, report.training_time, \
         report.comm_time, report.update_response, self.client_id), weights


#implement load model stuff!

#TODO WHY IS ACTION THE SAME??

def evaluate_policy(trainer, env, eval_episodes = 10):
        avg_reward = 0
        for i in range(eval_episodes):
            obs = env.reset()
            #print("obs in 129 in client evaluate policy", obs)
            done = False
            step_num = 0
            while not done:
                action = trainer.select_action(np.array(obs))
                #print("episode num i", i)
                #print("step num", step_num)
                #print("action in 135 in eval policy in client", action)
                #print("obs in 136 in eval policy in client", np.array(obs))
                obs, reward, done, _ = env.step(action)
                step_num += 1
                avg_reward += reward
        avg_reward /= eval_episodes
        print ("---------------------------------------")
        print ("Average Reward over the Evaluation Step: %f" % (avg_reward))
        print ("---------------------------------------")
        return avg_reward