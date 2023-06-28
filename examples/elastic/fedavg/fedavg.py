"""
A FedAvg session using modified MobilenetV3 model.
"""


from plato.clients.simple import Client
from fedavg_server import Server
from plato.datasources import wikitext2
from fedavg_trainer import Trainer
from transformer import Transformer


def main():
    """
    The FedAvg session.
    """
    ds = wikitext2.DataSource()
    print(len(ds.trainset), len(ds.testset))
    server = Server(model=Transformer, trainer=Trainer, datasource=wikitext2.DataSource)
    client = Client(model=Transformer, trainer=Trainer, datasource=wikitext2.DataSource)
    server.run(client)


if __name__ == "__main__":
    main()
