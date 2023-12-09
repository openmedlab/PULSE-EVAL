
import os
import random
import openai
from copy import deepcopy
os.environ['OPENAI_API_BASE'] = "https://api.emabc.xyz/v1"
api_key = ''
openai.api_key = api_key

from elo_utils import (
    construct_elo_inputs, 
    elo_evaluation, 
    call_evaluator
    )
from eval_utils import (
    acc_evaluation, 
    merge_table, 
    get_table_rank,
    write_to_md,
    add_avg_rank
    )



if __name__ == "__main__":
    random.seed(42)
    # Models for evaluation
    eval_models = [
        'ChatGPT',
        'GPT-4',
        'PULSE-Pro',
        'PULSE-OS',
        'Baichuan2',
        'ChatGLM3',
        'HuatuoGPT2',
        'QiZhenGPT',
        'BenTsao',
        'MING',
        'BianQue2',
        'DoctorGLM',
    ]
    
    # Datasets for evaluation
    eval_datasets = [
        'MedQA-USMLE',
        'MedQA-Mainland',
        'PromptCBLUE',
        'WebMedQA',
        'CheckupQA',
        'MedicineQA',
        'DialogSumm',
        'MedTriage',
    ]
    # Construct Elo inputs. 
    # If you have already run this, you should comment out this line.
    construct_elo_inputs(eval_models, eval_datasets, start_p_idx=0)
    
    # Call Evaluator API to get Elo results.
    # If you have already run this, you should comment out this line.
    call_evaluator(
        in_dir="eval/elo/elo_inputs",
        to_dir="eval/elo/elo_outputs",
        num_proc=50,
        )
    
    # Get evaluation results
    eval_datasets.remove("MedTriage")
    elo_table = elo_evaluation(eval_models, eval_datasets)
    acc_table = acc_evaluation(eval_models, ["MedTriage"])
    all_table = merge_table(elo_table.drop("ALL", axis=1), acc_table, ['MedTriage (F1)'])
    # score to rank
    datasets = eval_datasets + ['MedTriage (F1)']
    rank_table = get_table_rank(deepcopy(all_table), datasets)
    rank_table = add_avg_rank(rank_table, datasets)
    
    all_table = merge_table(all_table, rank_table, ['AVG Rank'])
    
    write_order = ["Model Name", "AVG Rank"] + datasets 
    write_to_md(all_table[write_order], to_path="score_table.md", sort_value="AVG Rank")
    write_to_md(rank_table[write_order], to_path="rank_table.md", sort_value="AVG Rank")
        