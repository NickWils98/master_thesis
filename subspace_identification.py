import numpy as np
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt
from debias_class import DebiasBert

# Function to load the pre-trained model and tokenizer
def load_model(model_type):
    tokenizer = AutoTokenizer.from_pretrained(model_type)
    model = AutoModel.from_pretrained(model_type)
    print(f"Loaded {model_type}")
    return tokenizer, model

# Function to calculate cosine similarity between two vectors
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# Function to get embeddings for a given text
def get_embeddings(tokenizer, model, text):
    word_embeddings_weight = model.get_input_embeddings().weight.data.numpy()
    inputs = tokenizer.encode(text, add_special_tokens=False)
    embeddings = word_embeddings_weight[inputs]

    # If multiple embeddings, average them
    if len(embeddings) > 1:
        embeddings = [np.mean(embeddings, axis=0)]

    return embeddings[0]

# Function to load gender word pairs from files
def load_gender_pairs(file_paths):
    gender_pairs = []
    gender_words = set()
    for file_path in file_paths:
        with open(file_path, 'r') as file:
            for line in file:
                words = line.strip().split()
                gender_pairs.append(words)
                gender_words.update(words)

    return gender_pairs, list(gender_words)

# Function to evaluate the debiasing by calculating the average cosine similarity
def evaluate_debiasing(tokenizer, model, gender_pairs):
    embedding_dict = {}

    # Get embeddings for each word in the gender pairs
    for pair in gender_pairs:
        for word in pair:
            if word not in embedding_dict:
                embedding_dict[word] = get_embeddings(tokenizer, model, word)

    # Calculate cosine similarity for each pair
    scores = [cosine_similarity(embedding_dict[pair[0]], embedding_dict[pair[1]]) for pair in gender_pairs]
    return np.mean(scores)

if __name__ == '__main__':
    # Load gender word pairs from files
    file_paths = []
    file_paths.append('data/generalized_swaps.txt')
    # file_paths.append('data/extra_gendered_words.txt')
    gender_pairs, gender_words = load_gender_pairs(file_paths)

    # Set the model type to be used
    # model_type = "bert-base-uncased"
    model_type = "gpt2"

    # Load the model and tokenizer
    tokenizer, model = load_model(model_type)
    debias_class = DebiasBert(tokenizer, debias_data=gender_words)

    score_run = []

    # Loop over different numbers of components for debiasing
    for num_components in range(0, 21):
        print(f"\nRun with {num_components} components")

        # Reload the model to reset embeddings
        tokenizer, model = load_model(model_type)

        # Debias the embeddings if the number of components is not zero
        if num_components != 0:
            debias_class.set_num_components(num_components)
            # debias_class.set_stepsize(num_components)
            debiased_embeddings = debias_class.debias_word_embeddings(model.get_input_embeddings())
            model.set_input_embeddings(debiased_embeddings)

        # Evaluate the debiased model
        score = evaluate_debiasing(tokenizer, model, gender_pairs)
        score_run.append(score)
        print(f"Score: {score}")

    # Print all scores
    print("Scores:", score_run)

    # Plot the results
    plt.figure()
    plt.plot(range(0, 21), score_run, marker='o', linestyle='-')
    plt.xlabel('Number of Dimensions')
    plt.ylabel('Average Cosine Similarity')
    plt.title('Effect of Number of Dimensions on the Vector Space')
    plt.savefig('plots\\Cosine_similarity_gpt2.png')
    plt.show()
