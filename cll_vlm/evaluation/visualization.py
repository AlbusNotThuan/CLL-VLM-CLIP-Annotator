import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class Visualization:
    @staticmethod
    def plot_distribution(scores, threshold):
        plt.figure(figsize=(7,5))
        plt.hist(scores, bins=50, color='skyblue', edgecolor='black')
        plt.axvline(threshold, color='red', linestyle='--', label=f'Threshold={threshold}')
        plt.legend(); plt.xlabel("Similarity score"); plt.ylabel("Count")
        plt.title("Distribution of similarity scores")
        plt.show()

    @staticmethod
    def plot_confusion_matrix(cm, class_names):
        plt.figure(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names, cmap="Blues")
        plt.xlabel("Predicted"); plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.show()