from helpers.embedding_evaluation.evaluate_similarity import EvaluationSimilarity
from helpers.embedding_evaluation.evaluate_feature_norm import EvaluationFeatureNorm
from helpers.embedding_evaluation.evaluate_concreteness import EvaluationConcreteness
import json

class Evaluation:

    def __init__(self, vocab_path=None, entity_subset=None, vocab=None, eval_conc=False, benchmark_subset=False):
        self.sim = EvaluationSimilarity(entity_subset=entity_subset, benchmark_subset=benchmark_subset)

        self.eval_conc = eval_conc
        #if self.eval_conc:
        #    self.conc = EvaluationConcreteness(entity_subset=None)

        self.fn = None
        #if (vocab_path is not None) or (vocab is not None):
        #    self.fn = EvaluationFeatureNorm(entity_subset=entity_subset, vocab_path=vocab_path, vocab=vocab)

    def evaluate(self, my_embedding):
        results = {}
        results["similarity"] = self.sim.evaluate(my_embedding)

        if self.fn is not None:
            results["feature_norm"] = self.fn.evaluate(my_embedding)

        if self.eval_conc:
            results["concreteness"] = self.conc.evaluate(my_embedding)
        return results

    def evaluate_to_file(self, my_embedding, file_path=None):
        """ save results to a json """
        results = self.evaluate(my_embedding)
        self.save_to_file(results, file_path)

    def save_to_file(self, results, file_path=None):
        assert file_path is not None
        with open(file_path, "w") as fp:
            json.dump(results, fp, sort_keys=True, indent=4)

    def save_summary_to_file(self, results, file_path=None):
        assert file_path is not None
        # get no items evaluated
        results = {k1: {k2: {k3: v3 if k3 != 'final_word_pair_list' else len(v3) for k3, v3 in v2.items()} for k2, v2 in v1.items()} for k1, v1 in results.items()}
        # get summary
        summary_results = {k1: {k2: {k3:v3 for k3, v3 in v2.items() if k3 == 'all_entities' or k3 == 'final_word_pair_list'} for k2, v2 in v1.items()} for k1, v1 in results.items()}
         
        with open(file_path, "w") as fp:
            json.dump(summary_results, fp, sort_keys=True, indent=4)

