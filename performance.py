from train import run_training, get_parser

MODELS_LIST = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "all-mpnet-base-v2",
    "all-distilroberta-v1",
    "all-MiniLM-L12-v2",
]

def eval_models_clf():
    parser = get_parser()
    args = parser.parse_args([])

    model_paths = []
    results_list = []
    for model_name in MODELS_LIST:
        print(f"Training model: {model_name}")
        args.model_name = model_name
        path, results = run_training(args)
        model_paths.append(path)
    
    return model_paths

def main():
    pass

if __name__ == "__main__":
    main()