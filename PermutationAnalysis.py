import numpy as np


def calculate_ranking_score(data):
    # Replace this function with your specific ranking metric calculation
    return np.argsort(data)


def permutation_test(observed_scores, num_permutations=1000):
    observed_ranking = calculate_ranking_score(observed_scores)
    num_observed_permutations = 0

    for _ in range(num_permutations):
        # Permute the data
        permuted_scores = np.random.permutation(observed_scores)

        # Calculate ranking for the permuted data
        permuted_ranking = calculate_ranking_score(permuted_scores)

        # Check if the observed ranking difference is greater than permuted
        if np.sum(observed_ranking != permuted_ranking) >= np.sum(
                observed_ranking != calculate_ranking_score(observed_scores)):
            num_observed_permutations += 1

    # Calculate p-value
    p_value = num_observed_permutations / num_permutations

    return p_value


# Example usage:
# Replace 'group1_scores' and 'group2_scores' with your actual data
group1_scores = np.array([10, 15, 8, 12, 18])
group2_scores = np.array([5, 8, 14, 9, 11])

observed_scores_difference = np.sum(calculate_ranking_score(group1_scores) != calculate_ranking_score(group2_scores))

p_value = permutation_test(np.concatenate([group1_scores, group2_scores]))

print(f"Observed Ranking Difference: {observed_scores_difference}")
print(f"P-Value: {p_value}")
