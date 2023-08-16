
# Commands Ran

## To create the LeetCode data

```bash
poetry run python llm_battle_ground/scripts/run_get_data.py

poetry run python llm_battle_ground/scripts/run_sample_leetcode.py
```

## To create the similarity data

### OpenAI models

```bash
poetry run python llm_battle_ground/scripts/run_generate_similarity.py --step-size=40 --num-input-examples=20 --num-output-examples=20 --buffer=10 --model=gpt-3.5-turbo-0613
```

```bash
poetry run python llm_battle_ground/scripts/run_generate_similarity.py --step-size=40 --num-input-examples=20 --num-output-examples=20 --buffer=10 --model=gpt-4-0613
```

## To create the completion data

```bash
poetry run python llm_battle_ground/scripts/run_generate_response.py --model=gpt-3.5-turbo-0613 --run-mode="vanilla-zero-shot" --in-file-name="leetcode_sampled.csv"
```

```bash
poetry run python llm_battle_ground/scripts/run_generate_response.py --model=gpt-4-0613 --run-mode="vanilla-zero-shot"  --in-file-name="leetcode_sampled.csv"
```

## To run the LeetCode Evaluation

```bash
poetry run python llm_battle_ground/scripts/run_leetcode_eval.py
```
