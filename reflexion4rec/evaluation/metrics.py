import numpy as np
import math
import heapq

# MAE
def mae(predicted, max_r, min_r, mae=True):
    total = 0
    for (r, p) in predicted:
        if p > max_r:
            p = max_r
        if p < min_r:
            p = min_r

        sub = p - r
        if mae:
            total += abs(sub)
        else:
            total += sub ** 2

    return total / len(predicted)

# RMSE
def rmse(predicted, max_r, min_r):
    mse = mae(predicted, max_r, min_r, False)
    return math.sqrt(mse)

# Precision
def precision_at_k(r, k):
    """Score is precision @ k
    Relevance is binary (nonzero is relevant).
    >>> r = [0, 0, 1]
    >>> precision_at_k(r, 1)
    0.0
    >>> precision_at_k(r, 2)
    0.0
    >>> precision_at_k(r, 3)
    0.33333333333333331
    >>> precision_at_k(r, 4)
    Traceback (most recent call last):
        File "<stdin>", line 1, in ?
    ValueError: Relevance score length < k
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Precision @ k
    Raises:
        ValueError: len(r) must be >= k
    """
    assert k >= 1
    r = np.asarray(r)[:k] != 0
    if r.size != k:
        raise ValueError("Relevance score length < k")
    return np.mean(r)


def recall_at_k(r, k, all_pos_num):
    r = np.asarray(r)[:k]
    return np.sum(r) / all_pos_num


def hit_at_k(r, k):
    r = np.asarray(r)[:k]
    if np.sum(r) > 0:
        return 1.0
    else:
        return 0.0


def dcg_at_k(r, k, method=1):
    """Score is discounted cumulative gain (dcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Example from
    http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
    >>> r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
    >>> dcg_at_k(r, 1)
    3.0
    >>> dcg_at_k(r, 1, method=1)
    3.0
    >>> dcg_at_k(r, 2)
    5.0
    >>> dcg_at_k(r, 2, method=1)
    4.2618595071429155
    >>> dcg_at_k(r, 10)
    9.6051177391888114
    >>> dcg_at_k(r, 11)
    9.6051177391888114
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    Returns:
        Discounted cumulative gain
    """
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError("method must be 0 or 1.")
    return 0.0


def ndcg_at_k(r, k, method=1):
    """Score is normalized discounted cumulative gain (ndcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Example from
    http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
    >>> r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
    >>> ndcg_at_k(r, 1)
    1.0
    >>> r = [2, 1, 2, 0]
    >>> ndcg_at_k(r, 4)
    0.9203032077642922
    >>> ndcg_at_k(r, 4, method=1)
    0.96519546960144276
    >>> ndcg_at_k([0], 1)
    0.0
    >>> ndcg_at_k([1], 2)
    1.0
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    Returns:
        Normalized discounted cumulative gain
    """
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.0
    return dcg_at_k(r, k, method) / dcg_max


def evaluate_once(topk_preds, groundtruth):
    """Evaluate one user performance.
    Args:
        topk_preds: list of <item_id>. length of the list is topK.
        groundtruth: list of <item_id>.
    Returns:
        dict of metrics.
    """
    gt_set = set(groundtruth)
    topk = len(topk_preds)
    rel = []
    for iid in topk_preds:
        if iid in gt_set:
            rel.append(1)
        else:
            rel.append(0)
    return {
        "precision@k": precision_at_k(rel, topk),
        "recall@k": recall_at_k(rel, topk, len(gt_set)),
        "ndcg@k": ndcg_at_k(rel, topk, 1),
        "hit@k": hit_at_k(rel, topk),
        "rel": rel,
    }


def evaluate_all(user_item_scores, groudtruth, topk=10):
    """Evaluate all user-items performance.
    Args:
        user_item_scores: dict with key = <item_id>, value = <user_item_score>.
                     Make sure larger score means better recommendation.
        groudtruth: dict with key = <user_id>, value = list of <item_id>.
        topk: int
    Returns:
    """
    avg_prec, avg_recall, avg_ndcg, avg_hit = 0.0, 0.0, 0.0, 0.0
    rs = []
    cnt = 0
    for uid in user_item_scores:
        # [Important] Use shuffle to break ties!!!
        ui_scores = list(user_item_scores[uid].items())
        np.random.shuffle(ui_scores)  # break ties
        # topk_preds = heapq.nlargest(topk, user_item_scores[uid], key=user_item_scores[uid].get)  # list of k <item_id>
        topk_preds = heapq.nlargest(topk, ui_scores, key=lambda x: x[1]) # list of k tuples
        topk_preds = [x[0] for x in topk_preds]  # list of k <item_id>
        # print(topk_preds, groudtruth[uid])
        result = evaluate_once(topk_preds, groudtruth[uid])
        avg_prec += result["precision@k"]
        avg_recall += result["recall@k"]
        avg_ndcg += result["ndcg@k"]
        avg_hit += result["hit@k"]
        rs.append(result["rel"])
        cnt += 1

    avg_prec = avg_prec / cnt
    avg_recall = avg_recall / cnt
    avg_ndcg = avg_ndcg / cnt
    avg_hit = avg_hit / cnt
    msg = "\nNDCG@{}\tReca@{}\tHR@{}\tPrec@{}".format(topk, topk, topk, topk)
    msg += "\n{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}".format(avg_ndcg, avg_recall, avg_hit, avg_prec)

    print(msg)
    res = {
        'ndcg': avg_ndcg,
        'recall': avg_recall,
        'precision': avg_prec,
        'hit': avg_hit,
    }
    return msg, res


def main():
    ui_scores = {
        1: {11: 3, 12: 4, 13: 5, 14: 6, 15: 7},
        # 2: {11: 3, 12: 4, 13: 5, 14: 6, 15: 7},
        # 3: {11: 3, 12: 4, 13: 5, 14: 6, 15: 7},
        # 4: {11: 3, 12: 4, 13: 5, 14: 6, 15: 7},
        # 5: {11: 3, 12: 4, 13: 5, 14: 6, 15: 7},
    }
    gt = {
        1: [11, 15],
        # 2: [12, 13],
        # 3: [11, 14],
        # 4: [12, 15],
        # 5: [11],
    }
    evaluate_all(ui_scores, gt, 5)


if __name__ == "__main__":
    main()