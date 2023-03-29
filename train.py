import argparse
import torch

from src.data import SQuAD
from src.bert import BiEncoder
from torch.utils.data import DataLoader


def main(data_path, nb_epochs, batch_size):
    # Load dataset
    squad = SQuAD(data_path)
    data_loader = DataLoader(squad, batch_size=batch_size, shuffle=True, pin_memory=True)
    
    # Train model
    model = BiEncoder()
    model.train_(data_loader, nb_epochs=nb_epochs, batch_size=batch_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Train BERT based retriever."
    )
    parser.add_argument(
        "data_path",
        metavar="data_path",
        type=str,
        help="The path of the data to use for training the retriever.",
    )
    parser.add_argument(
        "nb_epochs",
        metavar="nb_epochs",
        type=int,
        help="The number of epochs of the training process.",
    )
    parser.add_argument(
        "batch_size",
        metavar="batch_size",
        type=int,
        help="The size of the batch to use for training.",
    )
    args = parser.parse_args()
    
    main(args.data_path, args.nb_epochs, args.batch_size)