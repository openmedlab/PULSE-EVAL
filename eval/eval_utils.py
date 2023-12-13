from glob import glob
import json
from collections import OrderedDict, defaultdict
from datasets import load_dataset
import regex as re
from pandas import DataFrame
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
import numpy as np
import os

DATASET_LIST = []
GUIDE_LIST = []


def get_all_predict_data(all_predict_paths):
    all_predict_data = OrderedDict()
    for path in all_predict_paths:
        model_name = path.split("/")[-2]
        all_predict_data[model_name] = []
        with open(path, "r", encoding="utf8") as f:
            for line in f:
                line = line.rstrip()
                if len(line) == 0:
                    continue
                line = json.loads(line)
                all_predict_data[model_name].append(line)

    # check len
    d_len = None
    for item in all_predict_data.values():
        if d_len is None:
            d_len = len(item)
            
        assert d_len == len(item)

    # check q
    for idx in range(d_len):
        q = None
        for model_name in all_predict_data.keys():
            if q is None:
                q = all_predict_data[model_name][idx]['question']
            
            assert q == all_predict_data[model_name][idx]['question']

    print("load predict success")
    return all_predict_data
 
      

def get_table_rank(table: DataFrame, dataset_list):
    columns = table.columns
    datasets = []
    for col in columns:
        if col in dataset_list:
            datasets.append(col)
    for d in datasets:
        array = np.array(table[d])
        ranks = len(array) - array.argsort().argsort()
        ranks = ranks.tolist()
        
        # Check same score to be the same rank
        r_dict = dict(zip(ranks, range(len(table))))  # rank to colomn index
        
        r_cnt = 1
        pre_s = table.loc[r_dict[r_cnt], d]
        n_cnt = 1
        for rk in range(2, len(table) + 1):
            score = table.loc[r_dict[rk], d]
            if score != pre_s:
                # add new ranks
                # Change ranks from r_cnt to (r_cnt+n_cnt-1)
                ave_rank = r_cnt / n_cnt
                # ave_rank = rk - n_cnt - 1
                for i in range(n_cnt):
                    table.loc[r_dict[rk-i-1], d] = ave_rank
                    
                pre_s = score
                r_cnt = rk
                n_cnt = 1
            else:
                r_cnt += rk
                n_cnt += 1
                
        ave_rank = r_cnt / n_cnt
        # ave_rank = rk - n_cnt
        for i in range(n_cnt):
            table.loc[r_dict[rk-i], d] = ave_rank
            
        # for i in range(len(table)):
        #     inum = int(table.loc[i, d])
        #     if table.loc[i, d] == inum:
        #         table.loc[i, d] = inum
    return table
    
def acc_evaluation(
    selected_models,
    dataset_lists,
    predict_dir = 'eval/predicted/',
):
    statistic = defaultdict(dict)
    for dataset_name in dataset_lists:
        for model_name in selected_models:
            dataset = load_dataset(
                'json',
                data_files=f"{predict_dir + model_name}/{dataset_name}.jsonl", 
                split='train'
            )
            statistic[model_name][f"{dataset_name} (Acc)"] = acc_triage(dataset)
            p, r, f1 = f1_triage(dataset)
            statistic[model_name][f"{dataset_name} (P)"] = p
            statistic[model_name][f"{dataset_name} (R)"] = r
            statistic[model_name][f"{dataset_name} (F1)"] = f1
            
    table = DataFrame(
        [
            {
                'Model Name': k,
                **{dk: dv for dk, dv in v.items()}
            }
            for k, v in statistic.items()
        ]
    )
    return table


def acc_triage(dataset):
    acc_cnt = 0
    
    for item in dataset:
        refs = set(item['reference_answer'].replace(" ", "").split("|"))
        prds = set(item['predict_answer'].replace(" ", "").split("|"))
        acc_cnt += int(refs == prds)
        
    return acc_cnt / len(dataset)


def f1_triage(dataset):
    tp = 0  # ture predicted number 
    ap = 0  # all predicted number
    ag = 0  # all ground-truth number
    
    for item in dataset:
        refs = item['reference_answer'].replace(" ", "").split("|")
        prds = item['predict_answer'].replace(" ", "").split("|")

        ap += len(prds)
        ag += len(refs)
        
        for r in refs:
            tp += int(r in prds)
    p = tp / (ap + 1e-10)
    r = tp / (ag + 1e-10)
    f1 = (2 * p * r) / (p + r + 1e-10)
    
    return p, r, f1


def merge_table(elo_table, acc_table, merge_names=[]):
    for i in range(len(elo_table)):
        for j in range(len(acc_table)):
            if elo_table.loc[i, 'Model Name'] == acc_table.loc[j, 'Model Name']:
                for name in merge_names:
                    elo_table.loc[i, name] = acc_table.loc[j, name]
    return elo_table


def add_avg_rank(rank_table, datasets):
    for i in range(len(rank_table)):
        avg_rank = 0.0
        for dataset in datasets:
            avg_rank += rank_table.loc[i, dataset]
        rank_table.loc[i, 'AVG Rank'] = avg_rank / len(datasets)

    return rank_table


def write_to_md(
    table, 
    to_path="eval_table.md",
    sort_value=None, 
    ascending=True, 
    floatfmt='.02f', 
    ):
    if sort_value:
        table = table.sort_values(sort_value, ascending=ascending)
    md_table = table.to_markdown(index=False, floatfmt=floatfmt).replace(".00", "").replace(".50", ".5")
    
    if os.path.exists(to_path):
        os.remove(to_path)
    with open(to_path, 'w') as f:
        f.write(md_table)
    print(f"Write evaluation results to {to_path}.")