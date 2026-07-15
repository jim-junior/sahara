from __future__ import print_function
from collections import Counter
import string
import re
import argparse
import json
import sys


class SQuADEvaluator:
    """
    Official evaluation class for v1.1 of the SQuAD dataset.
    """

    def __init__(self, expected_version="1.1"):
        self.expected_version = expected_version


    def normalize_answer(self, s):
        """Lower text and remove punctuation, articles and extra whitespace."""
    
        def remove_articles(text):
            return re.sub(r"\b(a|an|the)\b", " ", text)
    
        def white_space_fix(text):
            return " ".join(text.split())
    
        def remove_punc(text):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)
    
        def lower(text):
            return text.lower()
    
        return white_space_fix(remove_articles(remove_punc(lower(s))))
    
    def f1_score(self, prediction, ground_truth):
        prediction_tokens = self.normalize_answer(prediction).split()
        ground_truth_tokens = self.normalize_answer(ground_truth).split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    def exact_match_score(self, prediction, ground_truth):
        return self.normalize_answer(prediction) == self.normalize_answer(ground_truth)

    def metric_max_over_ground_truths(self, metric_fn, prediction, ground_truths):
        scores_for_ground_truths = []
        for ground_truth in ground_truths:
            score = metric_fn(prediction, ground_truth)
            scores_for_ground_truths.append(score)
        return max(scores_for_ground_truths)

    def compute_score(self, gold_answers, predictions):
        f1 = exact_match = total = 0

        for ground_truths, prediction in zip(gold_answers, predictions):
            total += 1
            exact_match += self.metric_max_over_ground_truths(
                self.exact_match_score, prediction, ground_truths
            )
            f1 += self.metric_max_over_ground_truths(
                self.f1_score, prediction, ground_truths
            )

        exact_match = 100.0 * exact_match / total
        f1 = 100.0 * f1 / total

        return {'exact_match': exact_match, 'f1': f1}


# evaluator = SQuADEvaluator()
# gold = [["الجو جميل", "الطقس لطيف"], ["مدينة الرياض"]]
# preds = ["الجو جميل", "الرياض"]
# result = evaluator.compute_score(gold, preds)
# print(result)