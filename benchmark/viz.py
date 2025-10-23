import matplotlib.pyplot as plt
import numpy as np


def plot_hist(counts, nbins, xlabel, ylabel, yscale=None, title=None, figsize=(10,6), xlim=None):
    '''
    xlim: (a, b)
    '''
    _ = plt.figure(figsize=figsize)

    plt.hist(counts, bins=nbins)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if yscale:
        plt.yscale(yscale)
    if title:
        plt.title(title)
    if xlim:
        plt.xlim(xlim)
    plt.tight_layout()

def plot_bar_multiple(labels, counts_dict, xlabel, ylabel, title, figsize=None,rot=30):
    # labels_dict: common labels
    # counts_dict: key -> counts for each series
    xcoords = np.array(list(range(len(labels))))
    fig = plt.figure(figsize=figsize)

    # width of each bar
    width = 0.8 / len(counts_dict)
    offset = width / 2
    
    for ndx, (key, values) in enumerate(counts_dict.items()):
        plt.bar(xcoords-offset+ndx*width, values, width=width, align='center', label=key)

    plt.xticks(xcoords, labels, rotation=rot, ha='right')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=rot, ha='right')
    plt.legend()
    
    plt.tight_layout()

def plot_bar(labels, counts, xlabel, ylabel, title, figsize=None,rot=30):
    xcoords = list(range(len(labels)))
    fig = plt.figure(figsize=figsize)

    # ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    # ax.bar(xcoords, counts, align='center')
    plt.bar(xcoords, counts, align='center')
    plt.xticks(xcoords, labels, rotation=rot, ha='right')

    # ax.set_xticks(xcoords)
    # ax.set_xticklabels(labels)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=rot, ha='right')
    
    plt.tight_layout()