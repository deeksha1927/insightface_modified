import sys
import os
from os.path import join
import numpy as np
import pickle
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc
import argparse

def get_feat_dict(path):
    im_list = []
    with open(path + '.txt', 'r') as f:
        for i in f:
            im_list.append(i.rstrip())
    feats = np.load(path + '.npy')
    return im_list, feats

def process_scores(names, feats):
    genuine_scores = []
    impostor_scores = []
    length = len(names)

    for i in range(length):
        id_name, _ = names[i].split('/')[-2:]
        for j in range(i + 1, length):
            compare_id_name, _ = names[j].split('/')[-2:]

            if id_name == compare_id_name:  # Genuine pairs
                genuine_scores.append(np.dot(feats[i], feats[j]))
            else:  # Impostor pairs
                impostor_scores.append(np.dot(feats[i], feats[j]))

    return np.array(genuine_scores), np.array(impostor_scores)

def calculate_and_plot_roc(model_name, dataset, home, ax):
    features_path = f'{home}/features_ear/{model_name}/{dataset}/{dataset}'
    names, feats = get_feat_dict(features_path)
    
    assert len(names) == len(feats), f'{len(names)} {len(feats)}'

    genuine_scores, impostor_scores = process_scores(names, feats)

    # Combine scores and labels
    scores = np.concatenate([genuine_scores, impostor_scores])
    labels = np.concatenate([np.ones(len(genuine_scores)), np.zeros(len(impostor_scores))])

    # Calculate ROC curve and AUROC
    fpr, tpr, _ = roc_curve(labels, scores)
    auroc = auc(fpr, tpr)

    # Plot the ROC curve
    ax.plot(fpr, tpr, label=f'{model_name} ({dataset}, AUROC = {auroc:.4f})')
    return auroc

def main(model_names, datasets, home, save_dir):
    fig, ax = plt.subplots(figsize=(10, 8))

    for model_name in model_names:
        for dataset in datasets:
            auroc = calculate_and_plot_roc(model_name, dataset, home, ax)
            print(f"{model_name} ({dataset}): AUROC = {auroc:.4f}")

    # Add plot details
    ax.plot([0, 1], [0, 1], color='red', linestyle='--', label='Random Chance')
    ax.set_title('Receiver Operating Characteristic (ROC) Curves')
    ax.set_xlabel('False Positive Rate (FPR)')
    ax.set_ylabel('True Positive Rate (TPR)')
    ax.legend(loc='lower right')
    ax.grid()

    # Save the plot
    save_path = f'{home}/{save_dir}/combined_roc_curve.png'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    print(f'Saved combined ROC plot to {save_path}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Multiple ROC Curves")
    parser.add_argument('--model-names', nargs='+', type=str, required=True, help='List of model names')
    parser.add_argument('--datasets', nargs='+', type=str, required=True, help='List of dataset names')
    parser.add_argument('--home', type=str, default='/store01/flynn/darun', help='Home directory path')
    parser.add_argument('--save-dir', type=str, default='plots_ear', help='Directory to save the plots and results')

    args = parser.parse_args()
    main(args.model_names, args.datasets, args.home, args.save_dir)

