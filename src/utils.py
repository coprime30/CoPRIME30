import random
import logging

import torch
import numpy as np

from seqeval import metrics as seqeval_metrics
from sklearn import metrics as sklearn_metrics
from sklearn.metrics import confusion_matrix

# from src import KoBertTokenizer, HanBertTokenizer
from transformers import (
    ElectraConfig,
    ElectraTokenizer,
    ElectraForSequenceClassification,
    ElectraForTokenClassification,
    ElectraForQuestionAnswering,
)

#%%
CONFIG_CLASSES = {
    "koelectra-base": ElectraConfig,
}
#%%
TOKENIZER_CLASSES = {
    "koelectra-base": ElectraTokenizer,
}
#%%
MODEL_FOR_SEQUENCE_CLASSIFICATION = {
    "koelectra-base": ElectraForSequenceClassification,
}

# #%%
MODEL_FOR_TOKEN_CLASSIFICATION = {
    "koelectra-base": ElectraForTokenClassification,
}

#%%
def init_logger():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )


#%%
def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.no_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


#%%
def simple_accuracy(labels, preds):
    return (labels == preds).mean()


#%%
def acc_score(labels, preds):
    return {
        "acc": simple_accuracy(labels, preds),
    }


#%%
def f1_pre_rec(labels, preds):
    return {
        "3_precision": sklearn_metrics.precision_score(labels, preds, average="macro"),
        "4_recall": sklearn_metrics.recall_score(labels, preds, average="macro"),
        "2_f1": sklearn_metrics.f1_score(labels, preds, average="macro"),
        "5_cf": confusion_matrix(labels, preds),
        "1_accuracy": acc_score(labels, preds),
        "6_each_comparison": np.transpose(np.array([labels, preds])),
    }


#%%
def show_ner_report(labels, preds):
    return seqeval_metrics.classification_report(labels, preds, suffix=True)


#%%
def compute_metrics(labels, preds):
    assert len(preds) == len(labels)
    return f1_pre_rec(labels, preds)
