import time
import pickle
import argparse


def main(model_type, question):
    # Load retriever
    if model_type == "TF_IDF":
        with open('pretrained/tf_idf.pkl', 'rb') as f:
            retriever = pickle.load(f)
            print("\n- TF_IDF retriever successfully loaded.")
    
    elif model_type == "BM25":
        with open('pretrained/tf_idf.pkl', 'rb') as f:
            retriever = pickle.load(f)
            print("\n- BM25 retriever successfully loaded.")
    
    elif model_type == "BERT":
        with open('pretrained/tf_idf.pkl', 'rb') as f:
            retriever = pickle.load(f)
            print("\n- BERT retriever successfully loaded.")
    
    else:
        raise ValueError("Unexpected 'model_type' argument. Should be either 'TF_IDF', 'BM25' or 'BERT'.")
    
    # Retrieve context
    start = time.time()
    _, context = retriever.retrieve(question)
    retrieval_time = time.time() - start
    print("- Context retrieved in {:.2f}s: \n{}".format(retrieval_time, context))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Retrieve the context of a given question from SQuAD dataset."
    )
    parser.add_argument(
        "model_type",
        metavar="model_type",
        type=str,
        help="An string defining the model type to use. Can be either 'TF_IDF', 'BM25' or 'BERT'.",
    )
    parser.add_argument(
        "question",
        metavar="question",
        type=str,
        help="The question from SQuAD dataset as a string",
    )
    args = parser.parse_args()
    
    main(args.model_type, args.question)