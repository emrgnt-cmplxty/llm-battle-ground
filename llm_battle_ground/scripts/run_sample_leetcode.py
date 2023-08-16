import random
import os
from llm_battle_ground.constants import DATA_DIRECTORY
from llm_battle_ground.types import Datasets
import pandas as pd

DIFFICULTY_SAMPLE_RATES = {
    1: 0.25,  # Easy
    2: 0.50,  # Medium
    3: 1.0,  # Hard
}

if __name__ == "__main__":
    random.seed(0)

    leetcode_data_path = os.path.join(
        DATA_DIRECTORY, Datasets.LEETCODE_FULL.value
    )
    leetcode_data = pd.read_csv(leetcode_data_path)

    locs = []
    for index in range(len(leetcode_data)):
        difficulty = leetcode_data.iloc[index]["difficulty"]
        sample_rate = DIFFICULTY_SAMPLE_RATES.get(
            difficulty, 0.25
        )  # Use difficulty to determine sample rate, default to 25% if not-set (RARE)
        rnd_gen = random.random()

        if rnd_gen > sample_rate:
            continue
        # Storing the index if it meets the sampling criteria
        locs.append(index)

    # Creating a new DataFrame with the selected indices
    sampled_data = leetcode_data.iloc[locs]

    # Optional: You can save the sampled data to a new CSV file if needed
    output_path = os.path.join(DATA_DIRECTORY, Datasets.LEETCODE_SAMPLED.value)
    sampled_data.to_csv(output_path, index=False)

    print(f"Sampled data saved to {output_path}.")
    print(f"Total selected samples: {len(locs)}")
