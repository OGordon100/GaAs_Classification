import itertools

import numpy as np
from matplotlib import pyplot as plt
from sklearn import metrics
from tensorflow_core.python.keras.utils import to_categorical


def confusion_matrix(y_truth, y_pred, cats, cmap=None, normalize=True, title=None):
    cm = metrics.confusion_matrix(y_truth, y_pred)
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    if not cmap:
        cmap = plt.get_cmap('Blues')

    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title("Confusion Matrix")
    plt.colorbar()

    if cats:
        tick_marks = np.arange(len(cats))
        plt.xticks(tick_marks, cats, rotation=45)
        plt.yticks(tick_marks, cats)

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))

    if title:
        plt.title(title)

    return cm


def plot_model_history(model):
    for plot_metric in model.metrics_names:
        plt.figure()

        plt.plot(model.history.history[plot_metric])
        plt.plot(model.history.history[f'val_{plot_metric}'])
        plt.legend([plot_metric, f'val_{plot_metric}'])

        plt.xlabel("Epoch")
        plt.ylabel(plot_metric)


def assert_onehot(y):
    if np.array(y).ndim == 1:
        y = to_categorical(y)
    return y


def assert_indices(y):
    if np.array(y).ndim == 2:
        y = np.argmax(y, axis=1)
    return y


def ROC_one_vs_all(y_preds, y_truth, cats):
    fpr = {}
    tpr = {}
    tholds = {}
    roc_auc = {}

    y_preds = assert_onehot(y_preds)
    y_truth = assert_onehot(y_truth)

    plt.figure()

    for i, cat in enumerate(cats):
        fpr[cat], tpr[cat], tholds[cat] = (metrics.roc_curve(
            y_truth[:, i].astype(int), y_preds[:, i]))
        roc_auc[cat] = metrics.auc(fpr[cat], tpr[cat])
        plt.plot(fpr[cat], tpr[cat], label=f'{cat}, AUC = {roc_auc[cat]:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')

    plt.title('Receiver Operating Characteristic')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.legend()

    return tpr, fpr, tholds, roc_auc


def PR_one_vs_all(y_preds, y_truth, cats):
    prec = {}
    recall = {}
    tholds = {}
    pr_auc = {}

    y_preds = assert_onehot(y_preds)
    y_truth = assert_onehot(y_truth)

    plt.figure()

    for i, cat in enumerate(cats):
        prec[cat], recall[cat], tholds[cat] = (metrics.precision_recall_curve(
            y_truth[:, i].astype(int), y_preds[:, i]))
        pr_auc[cat] = metrics.auc(recall[cat], prec[cat])
        plt.plot(prec[cat], recall[cat], label=f'{cat}, AUC = {pr_auc[cat]:.2f}')

    plt.title('Precision-Recall')
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.legend()

    return recall, prec, tholds, pr_auc


def test_classifier(y_preds, y_truth, cats=None, average="weighted"):
    y_preds = assert_onehot(y_preds)
    y_truth = assert_onehot(y_truth)
    y_pred_arg = assert_indices(y_preds)
    y_true_arg = assert_indices(y_truth)

    if not cats:
        cats = np.arange(y_pred_arg.max())

    performance = {}
    performance_funcs = [metrics.accuracy_score, metrics.balanced_accuracy_score, metrics.confusion_matrix,
                         metrics.hamming_loss, metrics.matthews_corrcoef]
    weighted_performance_funcs = [metrics.f1_score, metrics.precision_score,
                                  metrics.recall_score, metrics.jaccard_score]

    for func in performance_funcs:
        performance[func.__name__] = func(y_pred=y_pred_arg, y_true=y_true_arg)
    for func in weighted_performance_funcs:
        performance[f"{func.__name__}_{average}"] = func(y_pred=y_pred_arg, y_true=y_true_arg, average=average)

    ROC_one_vs_all(y_preds=y_preds, y_truth=y_truth, cats=cats)
    PR_one_vs_all(y_preds=y_preds, y_truth=y_truth, cats=cats)
    confusion_matrix(y_pred=y_pred_arg, y_truth=y_true_arg, cats=cats)

    return performance
