import pickle
import argparse

from src.data import SQuAD
from src.tf_idf import TFIDF_Retriever
from src.bm25 import BM25_Retriever
from src.bert import BiEncoder, BERT_Retriever


def main(model_type, data_path):
    data = SQuAD(data_path)
    contexts, questions = data.contexts, data.questions
    
    if model_type == "TF_IDF":
        print("\nInitializing TF_IDF retriever and computing accuracy...")
        retriever = TFIDF_Retriever(contexts, questions)
        print("- TF_IDF retriever achieves: {:.2f}% of accuracy.".format(100*retriever.accuracy))
        
        with open('retrievers/tf_idf.pkl', 'wb+') as f:
            pickle.dump(retriever, f)
            print("- TF_IDF retriever successfully saved at: 'retrievers/tf_idf.pkl'")
    
    elif model_type == "BM25":
        print("\nInitializing BM25 retriever and computing accuracy...")
        retriever = BM25_Retriever(contexts, questions)
        print("- BM25 retriever achieves: {:.2f}% of accuracy.".format(100*retriever.accuracy))
        
        with open('retrievers/bm25.pkl', 'wb+') as f:
            pickle.dump(retriever, f)
            print("- BM25 retriever successfully saved at: '/retrievers/bm25.pkl'")
    
    elif model_type == "BERT":
        print("\nInitializing BERT retriever and computing accuracy...")
        model = BiEncoder()
        model.load_pretrained_model('pretrained/bert_epoch0.pth')
        retriever = BERT_Retriever(contexts, questions, bi_encoder=model)
        print("- BERT retriever achieves: {:.2f}% of accuracy.".format(100*retriever.accuracy))
        
        with open('retrievers/bert.pkl', 'wb+') as f:
            pickle.dump(retriever, f)
            print("- BERT retriever successfully saved at: '/retrievers/bert.pkl'")
    
    else:
        raise ValueError("Unexpected 'model_type' argument. Should be either 'TF_IDF' or 'BM25'.")


if __name__ == "__main__":    
    parser = argparse.ArgumentParser(
        description="Initialize TF_IDF, BM25 or BERT retriever and save it in pickle file."
    )
    parser.add_argument(
        "model_type",
        metavar="model_type",
        type=str,
        help="An string defining the model type to initialize. Can be either 'TF_IDF', 'BM25' or 'BERT'.",
    )
    parser.add_argument(
        "data_path",
        metavar="data_path",
        type=str,
        help="The path of the data to use for initialization of the retriever.",
    )
    args = parser.parse_args()
    
    main(args.model_type, args.data_path)