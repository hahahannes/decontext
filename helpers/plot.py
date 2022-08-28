import matplotlib.pyplot as plt

def get_ax(axes, list_index):
    n_rows = len(axes)
    n_cols = len(axes[0])
    counter = 0
    for row_index in range(n_rows):
        for col_index in range(n_cols):
            if counter == list_index:
                return axes[row_index, col_index]
            counter += 1

def set_style_and_font_size():
    #plt.style.use('seaborn')

    tex_fonts = {
        # Use LaTeX to write all text
        "text.usetex": True,
        "font.family": "serif",
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": 12,
        "font.size": 12,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10
    }

    plt.rcParams.update(tex_fonts)

def model_prettify(model):
    mapping = {
        'bert-base': 'BERT-base',
        'gpt-2': 'GPT-2',
        'bert-large': 'BERT-large',
        'gpt-2-medium': 'GPT-2-medium',
        'sbert_bert': 'Sentence-BERT',
        'sbert_distill_roberta': 'Sentence-Distill-RoBERTa',
        'w2v': 'Word2Vec',
        'glove': 'GloVe',
        'deconf': 'Deconflated'
    }
    return mapping[model]