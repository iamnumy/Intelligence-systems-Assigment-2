from surprise import SVD, Dataset, Reader, accuracy
from surprise.model_selection import train_test_split as surprise_split
from sklearn.model_selection import train_test_split as sklearn_split
import pandas as pd
from collections import defaultdict

# Dataset path configuration
ratings_path = '../data/ml-latest-small/ratings.csv'

# Load the ratings dataset
ratings_data_frame = pd.read_csv(ratings_path)

# Preprocess 'timestamp' column to datetime format for temporal analysis
ratings_data_frame['timestamp'] = ratings_data_frame['timestamp'].replace('964982224a', pd.NaT)
ratings_data_frame['timestamp'] = pd.to_datetime(ratings_data_frame['timestamp'], unit='s', errors='coerce')

# Data partitioning: Random Split
random_train_data, random_test_data = sklearn_split(ratings_data_frame, test_size=0.2, random_state=42)

# Data partitioning: Temporal Split
temporal_split_threshold = ratings_data_frame['timestamp'].quantile(0.8)
temporal_train_data = ratings_data_frame[ratings_data_frame['timestamp'] < temporal_split_threshold]
temporal_test_data = ratings_data_frame[ratings_data_frame['timestamp'] >= temporal_split_threshold]

# Surprise dataset preparation
surprise_reader = Reader(rating_scale=(0.5, 5.0))
surprise_data_random = Dataset.load_from_df(random_train_data[['userId', 'movieId', 'rating']], surprise_reader)
surprise_data_temporal = Dataset.load_from_df(temporal_train_data[['userId', 'movieId', 'rating']], surprise_reader)

# Model training and evaluation for both data splits
for split_label, surprise_dataset in [('Random Split', surprise_data_random),
                                      ('Temporal Split', surprise_data_temporal)]:
    print(f"\nEvaluating {split_label} \n")
    trainset, testset = surprise_split(surprise_dataset, test_size=0.2, random_state=42)
    svd_model = SVD()
    svd_model.fit(trainset)

    # Test set predictions
    predictions = svd_model.test(testset)

    # Root Mean Square Error (RMSE) computation
    calculated_rmse = accuracy.rmse(predictions)
    print(f"RMSE: {calculated_rmse}")

    # Precision and recall calculation
    def calculate_precision_recall(predictions, top_k=10, relevance_threshold=4.0):
        user_estimated_true_ratings = defaultdict(list)
        for user_id, _, true_rating, estimated_rating, _ in predictions:
            user_estimated_true_ratings[user_id].append((estimated_rating, true_rating))

        user_precisions = dict()
        user_recalls = dict()
        for user_id, user_ratings in user_estimated_true_ratings.items():
            user_ratings.sort(key=lambda x: x[0], reverse=True)
            relevant_items_count = sum((true_rating >= relevance_threshold) for (_, true_rating) in user_ratings)
            recommended_items_in_top_k = sum(
                (estimated_rating >= relevance_threshold) for (estimated_rating, _) in user_ratings[:top_k])
            relevant_and_recommended_in_top_k = sum(
                ((true_rating >= relevance_threshold) and (estimated_rating >= relevance_threshold)) for
                (estimated_rating, true_rating) in user_ratings[:top_k])

            user_precisions[
                user_id] = relevant_and_recommended_in_top_k / recommended_items_in_top_k if recommended_items_in_top_k != 0 else 0
            user_recalls[
                user_id] = relevant_and_recommended_in_top_k / relevant_items_count if relevant_items_count != 0 else 0

        return user_precisions, user_recalls

    calculated_precisions, calculated_recalls = calculate_precision_recall(predictions, top_k=10,
                                                                           relevance_threshold=4.0)

    # Average precision, recall, and F1 score computation
    average_precision = sum(precision for precision in calculated_precisions.values()) / len(calculated_precisions)
    average_recall = sum(recall for recall in calculated_recalls.values()) / len(calculated_recalls)
    f1_score = 2 * (average_precision * average_recall) / (average_precision + average_recall) if (
                                                                                                              average_precision + average_recall) != 0 else 0

    print(f"Precision: {average_precision}\nRecall: {average_recall}\nF1 Score: {f1_score}")
