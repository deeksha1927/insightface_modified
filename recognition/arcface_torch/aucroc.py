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

def main(model_name, dataset, home, save_dir):
    # Model and dataset configuration
    
    
   

    # Path to features
    features_path = f'{home}/features_ear/{model_name}/{dataset}/{dataset}'
    names, feats = get_feat_dict(features_path)
    assert len(names) == len(feats), f'{len(names)} {len(feats)}'
    length = len(names)

    # Initialize counters for genuine and impostor
    genuine_scores = []
    impostor_scores = []
    cl = 0
    cr = 0
    preds = {}

    # Process each image
    for i in range(length):
        path = names[i]
        id_name, im_name = path.split('/')[-2:]

        # Classify pairs as genuine or impostor based on folder name
        for j in range(i + 1, length):
            compare_path = names[j]
            compare_id_name, _ = compare_path.split('/')[-2:]

            # Check if both images are from the same folder (genuine) or different folders (impostor)
            if id_name == compare_id_name:  # Same folder name -> genuine
                genuine_scores.append(np.dot(feats[i], feats[j]))
            else:  # Different folder name -> impostor
                impostor_scores.append(np.dot(feats[i], feats[j]))

    # Convert scores to numpy arrays for further processing
    genuine_scores = np.array(genuine_scores)
    impostor_scores = np.array(impostor_scores)

    # Calculate ROC and AUROC
    scores = np.concatenate([genuine_scores, impostor_scores])
    labels = np.concatenate([np.ones(len(genuine_scores)), np.zeros(len(impostor_scores))])

    # Calculate the ROC curve
    fpr, tpr, thresholds = roc_curve(labels, scores)

    # Calculate the AUROC
    auroc = auc(fpr, tpr)
    print(f"AUROC: {auroc:.4f}")

    # Plot the ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUROC = {auroc:.4f})')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Random Chance')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.legend(loc='lower right')
    plt.grid()

    # Save the plot
    save_path = f'{home}/{save_dir}/{model_name}/{dataset}/aucroc/{dataset}-aucroc.png'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    with open(join(os.path.dirname(save_path), f'aucroc_{dataset}.txt'), 'w') as f:
        f.write('{:.3f}'.format(round(auroc, 3)))
    print('Saved plot and aucroc value.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot and Calculate D-Prime")
    parser.add_argument('--model-name', type=str, required=True, help='Model name')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--home', type=str, default='/store01/flynn/darun', help='Home directory path')
    parser.add_argument('--save-dir', type=str, default='plots_ear', help='Directory to save the plots and results')

    args = parser.parse_args()
    main(args.model_name, args.dataset, args.home, args.save_dir)

