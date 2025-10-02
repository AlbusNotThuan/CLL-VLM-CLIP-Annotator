import argparse
from dataset.cifar10 import CIFAR10Dataset
from models.clip_model import CLIPModel
from evaluation.evaluator import Evaluator
from evaluation.metrics import Metrics
from evaluation.visualization import Visualization

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="clip", help="Model type: clip | llava")
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    # Load dataset
    dataset = CIFAR10Dataset(root="../data/cifar10", train=True)

    # Init model
    if args.model == "clip":
        model = CLIPModel()
    else:
        raise NotImplementedError("Only CLIP implemented in main.py example")

    # Run evaluation
    evaluator = Evaluator(model, dataset, batch_size=args.batch_size, threshold=args.threshold)
    results = evaluator.run()

    # Compute metrics
    metrics = Metrics.compute(results["ground_truth"], results["predictions"])
    print(metrics)

    # Visualization
    Visualization.plot_distribution(results["scores"], args.threshold)
    Visualization.plot_confusion_matrix(metrics["confusion_matrix"], dataset.classes)

if __name__ == "__main__":
    main()