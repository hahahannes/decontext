### Decontextualized Transformer-based embeddings across world objects

#### Environment Setup

1. The code uses Python 3.8, [Pytorch 1.6.0](https://pytorch.org/), :hugs: [Transformers](https://github.com/huggingface/transformers) (Note that PyTorch 1.6.0 requires CUDA 10.2, if you want to decontextualize Transformer-based embeddings on a GPU)

2. Install PyTorch and :hugs: `transformers`: First run `pip install pytorch` (or `conda install pytorch torchvision -c pytorch`), and then `pip install transformers`. You can also install PyTorch and `transformers` in a single line with `pip install transformers[torch]`.

3. Install Python dependencies: `pip install -r requirements.txt`

4. Install [Cognival](https://github.com/hahahannes/cognival-cli/tree/decontext_requirements). First run `cd cognival-cli`, then `pip install -r requirements.txt`, then `python setup.py install`

#### Data for building embeddings
Scripts for pulling and processing datasets can be found in the `data/datasets` directory. For example, to pull the wikitext-2 dataset, run 
```
bash prepare-wikitext-2.sh
```

#### Annotating corpora with WordNet senses

We use Ares embeddings (see `get_ares.sh`) to determine the sense of a word in its context. Specifically, following the recommendations of the Ares authors, we choose the word sense who's Ares embedding is closest (cosine distance) to the contextualized embedding produced by BERT. To annotate a corpus with word senses, run the following command

```
python annotate_corpus.py --corpus_path /path/to/corpus/txt --ares_path /path/to/ares/embedding/txt/file --wordset_path /csv/with/words/to/annotate --out_path /pkl/output/file/name
```
where the `wordset_path` should be, e.g., the simlex999 word set. This will produce a pkl file (at `out_path`) which is a mapping from word -> wordsenses -> sentence index and string ranges for each word sense in the given corpus.

#### Creating embeddings from annotated corpus

Once a mapping has been created, we can use all (or a sample of) the contextual embeddings associated with a word sense to create a new embedding for that word sense.

To create new embeddings, run the following command

```
python extract_embeddings.py --corpus_path /path/to/corpus/txt --idx_path /path/to/mapping/pkl --out_path /output/dir
```
This will create an text file with the embeddings at `/output/dir`.  Additional parameters, such as language model, number of embeddings to aggregate over, and pooling function, can be set using flags. Run with the `--help` flag to see all options.

#### Running eval
* Add these lines to your `~/.bashrc`:
    * `export PYTHONPATH=$PYTHONPATH:path/to/embedding_evaluation/`
    * `export EMBEDDING_EVALUATION_DATA_PATH='path/to/embedding_evaluation/data/'`

##### Quick example
The global approach is given with those simple lines.
Your embeddings are in a `.txt` file that contains a word followed by it's embeddings, space separated.

```python
from embedding_evaluation.evaluate import Evaluation
from embedding_evaluation.load_embedding import load_embedding_textfile

# Load embeddings as a dictionnary {word: embed} where embed is a 1-d numpy array.
embeddings = load_embedding_textfile("path/to/my/embeddings.txt")

# Load and process evaluation benchmarks
evaluation = Evaluation() 
results = evaluation.evaluate(embeddings)
evaluation.save_summary_to_file(results, 'results_summary.json')
