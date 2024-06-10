import torch
import numpy as np
from sklearn.decomposition import PCA
from data.professions import get_all_professions
import copy
import matplotlib.pyplot as plt


class DebiasBert:
    def __init__(self, tokenizer, debias_data=None, num_components=10, stepsize=1):
        """
        Initialize the DebiasBert class with given parameters.

        :param tokenizer: The tokenizer used to convert words to tokens.
        :param debias_data: The data to be debiased. Default is professions data.
        :param num_components: The number of PCA components to retain.
        :param stepsize: The step size for PCA component selection.
        """
        self.debias_data = debias_data if debias_data is not None else get_all_professions()
        self.num_components = num_components
        self.stepsize = stepsize
        self.tokenizer = tokenizer

        # Define male and female words
        male_words = ['he', 'son', 'his', 'father', 'man', 'guy', 'John', 'boy', 'himself', 'male']
        female_words = ['she', 'daughter', 'her', 'mother', 'woman', 'gal', 'Mary', 'girl', 'herself', 'female']

        # Tokenize the words
        self.male_tokens = self._tokenize_words(male_words)
        self.female_tokens = self._tokenize_words(female_words)
        self.debias_data_tokens = self._tokenize_words(self.debias_data)

        self.dimension_list = []

    def set_num_components(self, num_components):
        """
        Set the number of PCA components to retain.

        :param num_components: Number of PCA components.
        """
        self.num_components = num_components

    def set_stepsize(self, stepsize):
        """
        Set the step size for PCA component selection.

        :param stepsize: Step size for PCA.
        """
        self.stepsize = stepsize

    def _tokenize_words(self, words):
        """
        Tokenize a list of words using the provided tokenizer.

        :param words: List of words to tokenize.
        :return: List of token IDs.
        """
        tokens = []
        for word in words:
            token_ids = self.tokenizer.encode(word, add_special_tokens=False)
            tokens.extend(token_ids)
        return tokens

    def get_gender_direction(self, embeddings):
        """
        Compute the gender direction vectors using PCA.

        :param embeddings: Word embeddings.
        :return: List of gender direction vectors.
        """
        embeddings2 = copy.deepcopy(embeddings)
        word_embeddings = embeddings2.weight.data.numpy()
        final_gender_bias_directions = []
        all_explained_variance = []

        while True:
            gender_words_male = [word_embeddings[token] for token in self.male_tokens]
            gender_words_female = [word_embeddings[token] for token in self.female_tokens]

            # matrix = [m-f for m, f in zip(gender_words_male, gender_words_female)]

            # Center the male and female word embeddings
            matrix = []
            for i in range(len(gender_words_male)):
                center = (gender_words_male[i] + gender_words_female[i]) / 2
                matrix.append(gender_words_male[i] - center)
                matrix.append(gender_words_female[i] - center)

            # Apply PCA to find principal components
            pca = PCA(n_components=self.stepsize)
            pca.fit(matrix)

            new_directions = []
            components = pca.components_

            for i in range(len(components)):
                if len(final_gender_bias_directions) < self.num_components:
                    component = components[i]
                    all_explained_variance.append(pca.explained_variance_[i])
                    print(pca.explained_variance_[i])
                    new_direction = component / np.linalg.norm(component)

                    # Orthogonalize new direction against existing directions
                    for prev_direction in final_gender_bias_directions + new_directions:
                        new_direction -= np.dot(new_direction, prev_direction) * prev_direction
                        new_direction /= np.linalg.norm(new_direction)

                    new_directions.append(new_direction)
                else:
                    break

            if new_directions:
                final_gender_bias_directions.extend(new_directions)
                word_embeddings = self.debias_gendered(word_embeddings, new_directions)
            else:
                break

        # self.plot_explained_variance(all_explained_variance)
        return final_gender_bias_directions

    def debias_word_embeddings(self, embeddings):
        """
        Debias neutral word embeddings using the computed gender directions.

        :param embeddings: Word embeddings.
        :return: Debiased embeddings.
        """
        gender_directions = self.get_gender_direction(embeddings)
        embeddings = self.debias_operation(embeddings, gender_directions)
        return embeddings

    def debias_full_word_embeddings(self, embeddings):
        """
        Debias the full set of word embeddings.

        :param embeddings: Word embeddings.
        :return: Debiased embeddings.
        """
        gender_directions = self.get_gender_direction(embeddings)
        embeddings_data = embeddings.weight.data.numpy()
        debiased_embeddings = embeddings_data.copy()
        debiased_embeddings = self.debias_gendered(debiased_embeddings, gender_directions)
        embeddings.weight.data = torch.tensor(debiased_embeddings)
        return embeddings

    def plot_explained_variance(self, all_explained_variance):
        """
        Plot the explained variance of each PCA dimension.

        :param all_explained_variance: List of explained variances for each dimension.
        """
        print("Explained Variance:", all_explained_variance)
        plt.figure()
        plt.plot(range(len(all_explained_variance)), all_explained_variance, marker='o', linestyle='-')
        plt.xlabel('Dimension Number')
        plt.ylabel('Explained Variance')
        plt.title('Explained Variance of each Dimension')
        plt.savefig('plots\\Explained_Variance.png')
        plt.show()

    def debias_gendered(self, word_embeddings, gender_directions):
        """
        Debias the word embeddings in the specified gender directions.

        :param word_embeddings: Word embeddings.
        :param gender_directions: Gender direction vectors.
        :return: Debiased word embeddings.
        """
        for direction in gender_directions:
            word_embeddings -= np.dot(word_embeddings, direction[:, np.newaxis]) * direction
        return word_embeddings

    def debias_operation(self, embeddings, gender_directions):
        """
        Debias the embeddings for the debias_data tokens.

        :param embeddings: Word embeddings.
        :param gender_directions: Gender direction vectors.
        :return: Debiased embeddings.
        """
        if not gender_directions:
            return embeddings

        embeddings_data = embeddings.weight.data.numpy()
        debiased_embeddings = embeddings_data.copy()

        for token in self.debias_data_tokens:
            embedding_vector = debiased_embeddings[token]
            for direction in gender_directions:
                embedding_vector -= np.dot(embedding_vector, direction) * direction
            debiased_embeddings[token] = embedding_vector

        embeddings.weight.data = torch.tensor(debiased_embeddings)
        return embeddings
