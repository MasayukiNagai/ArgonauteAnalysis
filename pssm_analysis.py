import numpy as np
import pandas as pd
from Bio import AlignIO
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import logomaker


aa1_to_index = {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6,
                'I': 7, 'K': 8, 'L': 9, 'M': 10, 'N': 11, 'P': 12,
                'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18,
                'Y': 19, 'X':20, 'Z': 21, '-': 22, 'B': 23}
# AA_ALPHABETS = "ACDEFGHIKLMNPQRSTVWY-"
color_dict = {
    'DEC': [.42, .16, .42],
    'PG': [.47, .47, 0.0],
    'MIWALFV': [.13, .35, .61],
    'NTSQ': [.25, .73, .28],
    'RK': [.74, .18, .12],
    'HY': [.09, .47, .46],
    '-': [0, 0, 0],
}

def calc_pssm_from_msa(msa, num_classes):
    msa2onehot = encode_msa2onehot(msa, num_classes)
    msas = torch.stack(list(msa2onehot.values()))
    pssm = calc_pssm(msas)
    return pssm


def encode_msa2onehot(path, num_classes):
    with open(path, "r") as f:
        alignment = AlignIO.read(f, "fasta")
    encoded_sequences = {}
    for record in alignment:
        x_numeric = numeric_encode(record.seq, max_index=num_classes - 1)
        x_one_hot = nn.functional.one_hot(
            x_numeric.long(), num_classes
        ).float()  # [N, L, C]
        encoded_sequences[record.id] = x_one_hot
    return encoded_sequences


def numeric_encode(seq, max_index=None):
    seq_numeric = np.array(
        [aa1_to_index[aa] for aa in str(seq).upper().replace(".", "-")]
    )
    if max_index is not None:
        seq_numeric[seq_numeric > max_index] = max_index
    return torch.from_numpy(seq_numeric)


def calc_pssm(data):
    freqs = torch.sum(data, dim=0)
    probs = freqs / torch.sum(freqs, dim=1, keepdims=True)
    probs = probs.detach().cpu().numpy()
    return probs


def plot_pssm(pssm, ALPHABETS, title=None):
    # pssm: [C, L]
    plt.figure(figsize=(20, 10))
    ax = sns.heatmap(pssm.T)
    if title is not None:
        plt.title(title)
    plt.xlabel("Position in Sequence")
    plt.ylabel("Amino Acid")
    ax.set_yticks(np.arange(len(ALPHABETS)) + 0.5)
    ax.set_yticklabels(list(ALPHABETS), rotation=0)


def calc_information_content(pssm_df, background_freq):
    # Calculate the information content for each position
    ic_df = pd.DataFrame(index=pssm_df.index)
    for column in pssm_df.columns:
        p_i = pssm_df[column]
        q_i = background_freq[column]
        ic = p_i.apply(lambda x: x * np.log2(x / q_i) if x > 0 else 0)
        ic_df[column] = ic
    # Sum across rows to get the total information content per position
    ic_df["Total IC"] = ic_df.sum(axis=1)
    return ic_df


def plot_logomaker(pssm_df, length_per_row=100, title=None):
    L = length_per_row
    num_rows = len(pssm_df) // L + 1

    fig, axs = plt.subplots(num_rows, 1, figsize=(24, 2 * num_rows))

    ylim = pssm_df.max().max()  # not ideal for probability

    for i in range(num_rows):
        logo = logomaker.Logo(
            pssm_df.iloc[(i) * L : (i + 1) * L, :],
            color_scheme=color_dict,
            vpad=0.1,
            width=0.8,
            ax=axs[i],
        )
        logo.style_xticks(anchor=0, spacing=5, rotation=45)
        logo.ax.set_ylim([0, ylim])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if title is not None:
        plt.suptitle(f"{title}", fontsize=18)
