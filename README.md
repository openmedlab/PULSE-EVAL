# PULSE-EVAL

## 1. 使用方法

## 1.1 目录结构

- data：原始测试数据集，每个数据集包含150条测试样例
- eval：测试库
  - elo: 存放三类elo数据
    - elo_data: elo比赛的参赛信息
    - elo_inputs: Evaluator的输入数据
    - elo_outputs: Evaluator的输出数据
  - predicted: 各模型的预测数据
  - 其余：代码文件

### 1.2 安装环境依赖包:

在conda环境中执行以下命令：

```
conda env create -f environment.yaml
```

### 1.3 评测使用方法

评测主要分为五个步骤：

- 数据集预测：预测数据集，将预测后的数据集按照目录 `predicted`中的文件和数据格式存放；
- 构建Elo输入数据：请见main.py文件中 `construct_elo_inputs`函数；
- Elo比赛评测：请见main.py文件中 `call_evaluator`函数，如使用GPT-4或ChatGPT作为Evaluator，需要制定api_key；
- 评测指标计算：Elo评测指标计算请见main.py文件中 `elo_evaluation`函数。其余Acc和F1指标计算，根据数据集有所不同，main.py中 `acc_evaluation`给出了导诊场景中使用的F1分数计算方法。
- 将评测结果计算排名，并输出，请见score_table.md和rank_table.md。

## 2. 数据集介绍

### 2.1 评测数据集

本评测使用四类公开数据集，并开源四类自建的不同医疗应用数据集。

|   **公开数据集**   |                             **评测维度/能力**                             |
| :----------------------: | :-----------------------------------------------------------------------------: |
|  **MedQA USMLE**  |   基于美国医师执照考试(USMLE)的多项选择题数据集。测试模型的英文医学考试能力。   |
| **MedQA Mainland** |           中国大陆医师考试选择题数据集。测试模型的中文医学考试能力。           |
|  **PromptCBLUE**  | 中文医疗场景NLP任务转化为基于提示的语言生成任务数据集。测试模型的下游任务能力。 |
|    **WebMedQA**    |          中文线上医疗问诊问答对话数据集。测试模型的中文医疗对话能力。          |

| **自建数据集** |                          **评测维度/能力**                          |
| :------------------: | :------------------------------------------------------------------------: |
| **MedTriage** | 根据用户信息给出导诊建议的数据集。测试模型在可变候选科室条件下的导诊能力。 |
| **DialogSumm** |        从医患对话中生成五史一诉的数据集。测试模型的长文本生成能力。        |
| **MedicineQA** |   给定标准参考文献时的用药咨询数据集。测试模型对长文本的理解和总结能力。   |
| **CheckupQA** |  体检场景下的数值类咨询数据集。测试模型对于医疗相关数值的理解和分析能力。  |

### 2.2 数据格式
单条数据格式如下：
```
{
  "type": "", 
  "question": "", 
  "reference_answer": "",
  "predict_answer": ""
}
```
其中type是数据集的名称，reference_answer是标准或参考回答，predict_answer是模型的回答。

## 3. 评测结果

### 3.1 评测数据表

说明：

- 数据集后无括号，表示计算Elo分数；
- 其余计算指标包括：Acc, BLEU, F1等
- 最高分数加粗显示

| **Model Name** | **组织-中文名称** | **Model Size** | **AVG Rank** | **MedQA USMLE** | **MedQA Mainland** | **Prompt CBLUE** | **Web MedQA** | **Checkup QA** | **Medicine QA** | **DialogSumm** | **MedTriage (F1)** |
| :------------------: | :---------------------: | :------------------: | :----------------: | :-------------------: | :----------------------: | :--------------------: | :-----------------: | :------------------: | :-------------------: | :------------------: | :----------------------: |
|   **GPT-4**   |    **OpenAI**    |          -          |   **1.25**   |    **1129**    |      **1117**      |     **1110**     |        1116        |         1096         |    **1098**    |    **1109**    |      **0.65**      |
| **PULSE-Pro** |   **上海AILab**   |          -          |   **1.75**   |         1089         |           1092           |          1088          |   **1119**   |    **1105**    |         1083         |         1096         |           0.63           |
|  **ChatGPT**  |    **OpenAI**    |          -          |   **4.00**   |         1086         |           1057           |          1064          |        1053        |         1020         |         1029         |         1080         |           0.43           |
|  **开源模型**  |                        |                      |                    |                      |                          |                        |                    |                      |                      |                      |                          |
|   **PULSE**   |   **上海AILab**   |         20B         |   **4.13**   |         1042         |           1024           |          1039          |        1059        |         1049         |         1069         |         1076         |           0.40           |
| **Baichuan2** | **百川智能-百川** |         13B         |   **4.50**   |         1024         |           1041           |          1065          |        1044        |         1062         |         1035         |         1069         |           0.33           |
|  **ChatGLM3**  |   **智谱&清华**   |          6B          |   **5.63**   |         1038         |           1062           |          997          |        1012        |         1003         |         1024         |         1021         |           0.06           |
| **HuatuoGPT2** |  **港中深-华佗**  |         13B         |   **7.75**   |          955          |           993           |          985          |         963         |         983         |         1003         |         980         |           0.01           |
| **QiZhenGPT** |   **浙大-启真**   |         13B         |   **8.19**   |          955          |           959           |          945          |         989         |         1039         |          932          |         921         |           0.00           |
|  **BenTsao**  |  **哈工大-本草**  |          7B          |   **8.75**   |          961          |           921           |          936          |         910         |         927         |          986          |         920         |           0.02           |
|  **BianQue2**  | **华南理工-扁鹊** |          6B          |  **10.13**  |          913          |           928           |          919          |         988         |         974         |          900          |         908         |           0.00           |
|    **MING**    |   **上交-明医**   |          7B          |  **10.69**  |          902          |           909           |          924          |         867         |         862         |          960          |         918         |           0.01           |
| **DoctorGLM** |    **上科大**    |          6B          |  **11.25**  |          906          |           896           |          930          |         879         |         880         |          880          |         905         |           0.00           |

### 3.2 全数据集排名榜单

| **Model Name** | **组织-中文名称** | **Model Size** | **AVG Rank** | **MedQA USMLE** | **MedQA Mainland** | **Prompt CBLUE** | **Web MedQA** | **Checkup QA** | **Medicine QA** | **DialogSumm** | **MedTriage (F1)** |
| :------------------: | :---------------------: | :------------------: | :----------------: | :-------------------: | :----------------------: | :--------------------: | :-----------------: | :------------------: | :-------------------: | :------------------: | :----------------------: |
|   **GPT-4**   |    **OpenAI**    |          -          |   **1.25**   |           1           |            1            |           1           |          2          |          2          |           1           |          1          |            1            |
| **PULSE-Pro** |   **上海AILab**   |          -          |   **1.75**   |           2           |            2            |           2           |          1          |          1          |           2           |          2          |            2            |
|  **ChatGPT**  |    **OpenAI**    |          -          |   **4.00**   |           3           |            4            |           4           |          4          |          6          |           5           |          3          |            3            |
|  **开源模型**  |                        |                      |                    |                      |                          |                        |                    |                      |                      |                      |                          |
|  **PULSE-OS**  |   **上海AILab**   |         20B         |   **4.13**   |           4           |            6            |           5           |          3          |          4          |           3           |          4          |            4            |
| **Baichuan2** | **百川智能-百川** |         13B         |   **4.50**   |           6           |            5            |           3           |          5          |          3          |           4           |          5          |            5            |
|  **ChatGLM3**  |   **智谱&清华**   |          6B          |   **5.63**   |           5           |            3            |           6           |          6          |          7          |           6           |          6          |            6            |
| **HuatuoGPT2** |  **港中深-华佗**  |         13B         |   **7.75**   |          8.5          |            7            |           7           |          9          |          8          |           7           |          7          |           8.5           |
| **QiZhenGPT** |   **浙大-启真**   |         13B         |   **8.19**   |          8.5          |            8            |           8           |          7          |          5          |          10          |          8          |            11            |
|  **BenTsao**  |  **哈工大-本草**  |          7B          |   **8.75**   |           7           |            10            |           9           |         10         |          10          |           8           |          9          |            7            |
|  **BianQue2**  | **华南理工-扁鹊** |          6B          |  **10.13**  |          10          |            9            |           12           |          8          |          9          |          11          |          11          |            11            |
|    **MING**    |   **上交-明医**   |          7B          |  **10.69**  |          12          |            11            |           11           |         12         |          12          |           9           |          10          |           8.5           |
| **DoctorGLM** |    **上科大**    |          6B          |  **11.25**  |          11          |            12            |           10           |         11         |          11          |          12          |          12          |            11            |

### 3.3 公开数据集排名榜单

| **Model Name** | MedQA USMLE | MedQA Mainland | Prompt CBLUE | Web MedQA | **AVG Rank** |
| :------------------: | :---------: | :------------: | :----------: | :-------: | :----------------: |
|   **GPT-4**   |      1      |       1       |      1      |     2     |   **1.25**   |
| **PULSE-Pro** |      2      |       2       |      2      |     1     |   **1.75**   |
|  **ChatGPT**  |      3      |       4       |      4      |     4     |   **3.75**   |
|  **PULSE-OS**  |      4      |       6       |      5      |     3     |   **4.50**   |
| **Baichuan2** |      6      |       5       |      3      |     5     |   **4.75**   |
|  **ChatGLM3**  |      5      |       3       |      6      |     6     |   **5.00**   |
| **HuatuoGPT2** |     8.5     |       7       |      7      |     9     |   **7.88**   |
| **QiZhenGPT** |     8.5     |       8       |      8      |     7     |   **7.88**   |
|  **BenTsao**  |      7      |       10       |      9      |    10    |   **9.00**   |
|  **BianQue2**  |     10     |       9       |      12      |     8     |   **9.75**   |
| **DoctorGLM** |     11     |       12       |      10      |    11    |  **11.00**  |
|    **MING**    |     12     |       11       |      11      |    12    |  **11.50**  |

### 3.4 自建数据集排名榜单

| **Model Name** | MedQA USMLE | MedQA Mainland | Prompt CBLUE | Web MedQA | **AVG Rank** |
| :------------------: | :---------: | :------------: | :----------: | :-------: | :----------------: |
|   **GPT-4**   |      1      |       1       |      1      |     2     |   **1.25**   |
| **PULSE-Pro** |      2      |       2       |      2      |     1     |   **1.75**   |
|  **ChatGPT**  |      3      |       4       |      4      |     4     |   **3.75**   |
|  **PULSE-OS**  |      4      |       6       |      5      |     3     |   **4.50**   |
| **Baichuan2** |      6      |       5       |      3      |     5     |   **4.75**   |
|  **ChatGLM3**  |      5      |       3       |      6      |     6     |   **5.00**   |
| **HuatuoGPT2** |     8.5     |       7       |      7      |     9     |   **7.88**   |
| **QiZhenGPT** |     8.5     |       8       |      8      |     7     |   **7.88**   |
|  **BenTsao**  |      7      |       10       |      9      |    10    |   **9.00**   |
|  **BianQue2**  |     10     |       9       |      12      |     8     |   **9.75**   |
| **DoctorGLM** |     11     |       12       |      10      |    11    |  **11.00**  |
|    **MING**    |     12     |       11       |      11      |    12    |  **11.50**  |

### 3.5 数据集分类排名榜单

| **Model Name** | **公开集排名** | **自建集排名** | **平均排名** |
| :------------------: | :------------------: | :------------------: | :----------------: |
|   **GPT-4**   |          1          |          1          |         1         |
| **PULSE-Pro** |          2          |          2          |         2         |
|  **ChatGPT**  |          3          |          4          |         3         |
|  **PULSE-OS**  |          4          |          3          |         4         |
| **Baichuan2** |          5          |          5          |         5         |
|  **ChatGLM3**  |          6          |          6          |         6         |
| **HuatuoGPT2** |          7          |          7          |         7         |
| **QiZhenGPT** |          8          |          8          |         8         |
|  **BenTsao**  |          9          |          9          |         9         |
|  **BianQue2**  |          10          |          11          |         10         |
|    **MING**    |          12          |          10          |         11         |
| **DoctorGLM** |          11          |          12          |         12         |

## 4. 说明

本评测中，各模型的调用方法如下表所示。Elo评测中，采用"gpt-4-1106-preview"为Evaluator，测试结果获取日期为2023年12月7日。

|    **模型**    |                                                                                      **说明**                                                                                      |
| :------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| **PULSE-Pro** |                                                                                     使用模型fp16权重                                                                                     |
|  **PULSE-OS**  |                                                                                     使用模型fp16权重                                                                                     |
|   **GPT-4**   |                                                                          使用OpenAI的API："gpt-4-1106-preview"                                                                          |
|  **ChatGPT**  |                                                                          使用OpenAI的API："gpt-3.5-turbo-1106"                                                                          |
| **Baichuan2** |                             使用模型权重预测：[https://huggingface.co/baichuan-inc/Baichuan2-13B-Chat](https://huggingface.co/baichuan-inc/Baichuan2-13B-Chat)                             |
|  **ChatGLM3**  |                                       使用模型权重预测：[https://huggingface.co/THUDM/chatglm3-6b-32k](https://huggingface.co/THUDM/chatglm3-6b-32k)                                       |
| **HuatuoGPT2** |                                                          使用官方网站预测：[https://www.huatuogpt.cn/](https://www.huatuogpt.cn/)                                                          |
| **QiZhenGPT** |           使用基模型[CaMA-13B](https://github.com/zjunlp/CaMA)和官方权重"QiZhen-CaMA-13B-Checkpoint-12400"：[https://github.com/CMKRG/QiZhenGPT](https://github.com/CMKRG/QiZhenGPT)           |
|  **BenTsao**  |        使用官方基模型[活字1.0](https://github.com/HIT-SCIR/huozi)和LoRA权重。[https://github.com/SCIR-HI/Huatuo-Llama-Med-Chinese](https://github.com/SCIR-HI/Huatuo-Llama-Med-Chinese)        |
|  **BianQue2**  |                                           使用官方权重预测：[https://huggingface.co/scutcyr/BianQue-2](https://huggingface.co/scutcyr/BianQue-2)                                           |
|    **MING**    |                                           使用官方权重预测：[https://huggingface.co/BlueZeros/MING-7B](https://huggingface.co/BlueZeros/MING-7B)                                           |
| **DoctorGLM** | 使用基模型ChatGLM和官方[ptuning\_weight](https://pan.baidu.com/s/1Yf56egVGwI0XN2iOLcEGSQ?pwd=r4p0)权重：[https://github.com/xionghonglin/DoctorGLM](https://github.com/xionghonglin/DoctorGLM) |
