---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:100
- loss:MultipleNegativesRankingLoss
widget:
- source_sentence: What trend was observed in job layoffs as measured by initial claims
    for unemployment insurance benefits before the pandemic?
  sentences:
  - "options-implied modal path, likely reflecting investors’ perception that risks\
    \ were tilted toward more \n \n1 The Federal Open Market Committee is referenced\
    \ as the “FOMC” and the “Committee” in these minutes; the Board of"
  - "supply expected to decrease as a result of the September tax date before increasing\
    \ again, the staff \nassessed that the decline in ON RRP usage might slow over\
    \ the coming intermeeting period before"
  - "before the pandemic.  Job layoffs, as measured by initial claims for unemployment\
    \ insurance benefits, \nremained low through August.  Measures of nominal labor\
    \ compensation continued to decelerate."
- source_sentence: What did the manager mention about the concentration of ON RRP
    usage?
  sentences:
  - "quarter.  Both measures were well below their pace from a year earlier.  \nReal\
    \ GDP rose solidly, on balance, over the first half of the year.  Real private\
    \ domestic final purchases"
  - "capital goods, continued to grow at a robust pace in July. \nReal GDP growth\
    \ in foreign economies stepped down in the second quarter, and recent economic"
  - "resuming later this year.  The manager also added that, with the concentration\
    \ of ON RRP usage in a \nsmall number of fund complexes, there was an increased\
    \ risk that idiosyncratic allocation decisions"
- source_sentence: How did broad equity prices perform by the end of the period?
  sentences:
  - "After growing at a tepid pace in the second quarter, real exports of goods moved\
    \ down in July, led by \ndeclines in exports of autos and industrial supplies.\
    \  By contrast, real imports of goods, especially of"
  - "before the pandemic.  Job layoffs, as measured by initial claims for unemployment\
    \ insurance benefits, \nremained low through August.  Measures of nominal labor\
    \ compensation continued to decelerate."
  - "yields over the period was primarily attributable to lower expected real yields,\
    \ but measures of \ninflation compensation declined as well.  Broad equity prices\
    \ finished the period modestly higher,"
- source_sentence: Which types of indexes experienced large moves during the episode
    of elevated market volatility?
  sentences:
  - "respectively.  The market-implied expectations for year-end policy rates fell\
    \ over the period for most \ncentral banks in AFEs, although by a smaller amount\
    \ than they did for the Federal Reserve,"
  - "The manager also discussed the brief episode of elevated market volatility in\
    \ early August.  That \nepisode saw some large moves in U.S. and foreign equity\
    \ indexes, equity-implied volatilities, the"
  - "remained stable over the intermeeting period.  In secured funding markets, rates\
    \ on overnight \nrepurchase agreements (repo) were higher than a few months earlier\
    \ amid large issuance of Treasury"
- source_sentence: What does PDFP comprise of, and why is it considered to provide
    a better signal than GDP?
  sentences:
  - "modal path, and was unchanged thereafter.  Balance sheet expectations in the\
    \ surveys were little \nchanged from July.  Most survey respondents did not appear\
    \ to be concerned about an economic"
  - "The manager also discussed the brief episode of elevated market volatility in\
    \ early August.  That \nepisode saw some large moves in U.S. and foreign equity\
    \ indexes, equity-implied volatilities, the"
  - "(PDFP)—which comprises PCE and private fixed investment and which often provides\
    \ a better signal \nthan GDP of underlying economic momentum—posted a stronger\
    \ first-half increase than GDP, and"
pipeline_tag: sentence-similarity
library_name: sentence-transformers
metrics:
- cosine_accuracy@1
- cosine_accuracy@3
- cosine_accuracy@5
- cosine_accuracy@10
- cosine_precision@1
- cosine_precision@3
- cosine_precision@5
- cosine_precision@10
- cosine_recall@1
- cosine_recall@3
- cosine_recall@5
- cosine_recall@10
- cosine_ndcg@10
- cosine_mrr@10
- cosine_map@100
model-index:
- name: SentenceTransformer
  results:
  - task:
      type: information-retrieval
      name: Information Retrieval
    dataset:
      name: Unknown
      type: unknown
    metrics:
    - type: cosine_accuracy@1
      value: 0.94
      name: Cosine Accuracy@1
    - type: cosine_accuracy@3
      value: 0.99
      name: Cosine Accuracy@3
    - type: cosine_accuracy@5
      value: 1.0
      name: Cosine Accuracy@5
    - type: cosine_accuracy@10
      value: 1.0
      name: Cosine Accuracy@10
    - type: cosine_precision@1
      value: 0.94
      name: Cosine Precision@1
    - type: cosine_precision@3
      value: 0.33000000000000007
      name: Cosine Precision@3
    - type: cosine_precision@5
      value: 0.19999999999999996
      name: Cosine Precision@5
    - type: cosine_precision@10
      value: 0.09999999999999998
      name: Cosine Precision@10
    - type: cosine_recall@1
      value: 0.94
      name: Cosine Recall@1
    - type: cosine_recall@3
      value: 0.99
      name: Cosine Recall@3
    - type: cosine_recall@5
      value: 1.0
      name: Cosine Recall@5
    - type: cosine_recall@10
      value: 1.0
      name: Cosine Recall@10
    - type: cosine_ndcg@10
      value: 0.9758532532593068
      name: Cosine Ndcg@10
    - type: cosine_mrr@10
      value: 0.9675
      name: Cosine Mrr@10
    - type: cosine_map@100
      value: 0.9675
      name: Cosine Map@100
---

# SentenceTransformer

This is a [sentence-transformers](https://www.SBERT.net) model trained. It maps sentences & paragraphs to a 384-dimensional dense vector space and can be used for retrieval.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
<!-- - **Base model:** [Unknown](https://huggingface.co/unknown) -->
- **Maximum Sequence Length:** 256 tokens
- **Output Dimensionality:** 384 dimensions
- **Similarity Function:** Cosine Similarity
- **Supported Modality:** Text
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/huggingface/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'transformer_task': 'feature-extraction', 'modality_config': {'text': {'method': 'forward', 'method_output_name': 'last_hidden_state'}}, 'module_output_name': 'token_embeddings', 'architecture': 'BertModel'})
  (1): Pooling({'embedding_dimension': 384, 'pooling_mode': 'mean', 'include_prompt': True})
  (2): Normalize({})
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```
Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
queries = [
    'What does PDFP comprise of, and why is it considered to provide a better signal than GDP?',
]
documents = [
    '(PDFP)—which comprises PCE and private fixed investment and which often provides a better signal \nthan GDP of underlying economic momentum—posted a stronger first-half increase than GDP, and',
    'The manager also discussed the brief episode of elevated market volatility in early August.  That \nepisode saw some large moves in U.S. and foreign equity indexes, equity-implied volatilities, the',
    'modal path, and was unchanged thereafter.  Balance sheet expectations in the surveys were little \nchanged from July.  Most survey respondents did not appear to be concerned about an economic',
]
query_embeddings = model.encode_query(queries)
document_embeddings = model.encode_document(documents)
print(query_embeddings.shape, document_embeddings.shape)
# [1, 384] [3, 384]

# Get the similarity scores for the embeddings
similarities = model.similarity(query_embeddings, document_embeddings)
print(similarities)
# tensor([[0.7746, 0.0375, 0.0848]])
```
<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

## Evaluation

### Metrics

#### Information Retrieval

* Evaluated with [<code>InformationRetrievalEvaluator</code>](https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html#sentence_transformers.sentence_transformer.evaluation.InformationRetrievalEvaluator)

| Metric              | Value      |
|:--------------------|:-----------|
| cosine_accuracy@1   | 0.94       |
| cosine_accuracy@3   | 0.99       |
| cosine_accuracy@5   | 1.0        |
| cosine_accuracy@10  | 1.0        |
| cosine_precision@1  | 0.94       |
| cosine_precision@3  | 0.33       |
| cosine_precision@5  | 0.2        |
| cosine_precision@10 | 0.1        |
| cosine_recall@1     | 0.94       |
| cosine_recall@3     | 0.99       |
| cosine_recall@5     | 1.0        |
| cosine_recall@10    | 1.0        |
| **cosine_ndcg@10**  | **0.9759** |
| cosine_mrr@10       | 0.9675     |
| cosine_map@100      | 0.9675     |

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 100 training samples
* Columns: <code>sentence_0</code> and <code>sentence_1</code>
* Approximate statistics based on the first 100 samples:
  |          | sentence_0                                                                         | sentence_1                                                                         |
  |:---------|:-----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|
  | type     | string                                                                             | string                                                                             |
  | modality | text                                                                               | text                                                                               |
  | details  | <ul><li>min: 12 tokens</li><li>mean: 19.22 tokens</li><li>max: 38 tokens</li></ul> | <ul><li>min: 15 tokens</li><li>mean: 39.38 tokens</li><li>max: 59 tokens</li></ul> |
* Samples:
  | sentence_0                                                                                                              | sentence_1                                                                                                                                                                                                                                                     |
  |:------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>What was the date of the joint meeting of the Federal Open Market Committee and the Board of Governors?</code>    | <code>1 <br> <br>Minutes of the Federal Open Market <br>Committee <br>September 17–18, 2024 <br>A joint meeting of the Federal Open Market Committee and the Board of Governors of the Federal</code>                                                          |
  | <code>Which two entities held a meeting on September 17–18, 2024?</code>                                                | <code>1 <br> <br>Minutes of the Federal Open Market <br>Committee <br>September 17–18, 2024 <br>A joint meeting of the Federal Open Market Committee and the Board of Governors of the Federal</code>                                                          |
  | <code>What were the dates and times of the Reserve System meeting held in the offices of the Board of Governors?</code> | <code>Reserve System was held in the offices of the Board of Governors on Tuesday, September 17, 2024, <br>at 10:30 a.m. and continued on Wednesday, September 18, 2024, at 9:00 a.m.1 <br>Developments in Financial Markets and Open Market Operations</code> |
* Loss: [<code>MultipleNegativesRankingLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss) with these parameters:
  ```json
  {
      "scale": 20.0,
      "similarity_fct": "cos_sim",
      "gather_across_devices": false,
      "directions": [
          "query_to_doc"
      ],
      "partition_mode": "joint",
      "hardness_mode": null,
      "hardness_strength": 0.0
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 20
- `num_train_epochs`: 5
- `per_device_eval_batch_size`: 20
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `per_device_train_batch_size`: 20
- `num_train_epochs`: 5
- `max_steps`: -1
- `learning_rate`: 5e-05
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: None
- `warmup_steps`: 0
- `optim`: adamw_torch_fused
- `optim_args`: None
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `optim_target_modules`: None
- `gradient_accumulation_steps`: 1
- `average_tokens_across_devices`: True
- `max_grad_norm`: 1
- `label_smoothing_factor`: 0.0
- `bf16`: False
- `fp16`: False
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `use_cache`: False
- `neftune_noise_alpha`: None
- `torch_empty_cache_steps`: None
- `auto_find_batch_size`: False
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `include_num_input_tokens_seen`: no
- `log_level`: passive
- `log_level_replica`: warning
- `disable_tqdm`: False
- `project`: huggingface
- `trackio_space_id`: None
- `trackio_bucket_id`: None
- `trackio_static_space_id`: None
- `per_device_eval_batch_size`: 20
- `prediction_loss_only`: True
- `eval_on_start`: False
- `eval_do_concat_batches`: True
- `eval_use_gather_object`: False
- `eval_accumulation_steps`: None
- `include_for_metrics`: []
- `batch_eval_metrics`: False
- `save_only_model`: False
- `save_on_each_node`: False
- `enable_jit_checkpoint`: False
- `push_to_hub`: False
- `hub_private_repo`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_always_push`: False
- `hub_revision`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `restore_callback_states_from_checkpoint`: False
- `full_determinism`: False
- `seed`: 42
- `data_seed`: None
- `use_cpu`: False
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `dataloader_prefetch_factor`: None
- `remove_unused_columns`: True
- `label_names`: None
- `train_sampling_strategy`: random
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `ddp_static_graph`: None
- `ddp_backend`: None
- `ddp_timeout`: 1800
- `fsdp`: []
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `deepspeed`: None
- `debug`: []
- `skip_memory_metrics`: True
- `do_predict`: False
- `resume_from_checkpoint`: None
- `warmup_ratio`: None
- `local_rank`: -1
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin
- `router_mapping`: {}
- `learning_rate_mapping`: {}

</details>

### Training Logs
| Epoch | Step | cosine_ndcg@10 |
|:-----:|:----:|:--------------:|
| 1.0   | 5    | 0.9598         |
| 2.0   | 10   | 0.9685         |
| 3.0   | 15   | 0.9685         |
| 4.0   | 20   | 0.9685         |
| 5.0   | 25   | 0.9759         |


### Training Time
- **Training**: 12.7 seconds

### Framework Versions
- Python: 3.11.9
- Sentence Transformers: 5.5.0
- Transformers: 5.8.0
- PyTorch: 2.11.0
- Accelerate: 1.13.0
- Datasets: 4.8.5
- Tokenizers: 0.22.2

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

#### MultipleNegativesRankingLoss
```bibtex
@misc{oord2019representationlearningcontrastivepredictive,
      title={Representation Learning with Contrastive Predictive Coding},
      author={Aaron van den Oord and Yazhe Li and Oriol Vinyals},
      year={2019},
      eprint={1807.03748},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/1807.03748},
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->