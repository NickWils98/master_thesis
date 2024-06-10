import json
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM, AutoModelForSequenceClassification

import run_glue
from debias_class import DebiasBert
from metrics import lpbs, crowspairs, disco, seat


def load_model(model_type):
    """Load the tokenizer, masked language model, and base model for a given model type."""
    tokenizer = AutoTokenizer.from_pretrained(model_type)
    model_masked = AutoModelForMaskedLM.from_pretrained(model_type)
    model = AutoModel.from_pretrained(model_type)
    print(f"Loaded {model_type}")
    return tokenizer, model, model_masked


def create_directory_if_not_exists(directory):
    """Create a directory if it does not already exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)


def run_debiasing_experiment(range_list, filename1, attribute_template, target_template, plotbase, plotname):
    """Run the debiasing experiment for different dimensions and save the results."""
    lpbs_score = []
    lpbs_target_score = []
    seat_score = []
    disco_score = []

    for i in range_list:
        print(f"run {i}")
        tokenizer, model, model_masked = load_model("bert-base-uncased")

        if i > 0:
            debiaser = DebiasBert(tokenizer, None, i, 1)
            model.set_input_embeddings(debiaser.debias_word_embeddings(model.get_input_embeddings()))

            debiaser2 = DebiasBert(tokenizer, None, i, 1)
            model_masked.set_input_embeddings(
            debiaser2.debias_word_embeddings(model_masked.get_input_embeddings()))

        model.eval()
        model_masked.eval()

        results = {}
        with torch.no_grad(), open(f"{filename1}{i}.json", 'w') as file:
            print("Start lbps\n")
            score = lpbs.lpbs_test(tokenizer, model_masked, sent_debias=False)
            results["LPBS"] = score
            lpbs_score.append(score[0])
            lpbs_target_score.append(score[2])
            print(score)
            json.dump({"metrics": results}, file)

            print("Start disco\n")
            score = disco.disco_test(tokenizer, model_masked, sent_debias=False)
            results["DISCO"] = score
            print(score)
            disco_score.append(score)
            json.dump({"metrics": results}, file)

            print("Start seat\n")
            seat_results = seat.seat_test(attribute_template, target_template, tokenizer, model)
            score = np.fromiter(seat_results.values(), dtype=float).mean()
            results["SEAT"] = score
            print(score)
            seat_score.append(score)

            print("Start Crows\n")
            score = crowspairs.evaluate(tokenizer, model_masked)
            results["CROWS-PAIRS"] = score
            print(score)
            json.dump({"metrics": results}, file)

    plot_results(range_list, lpbs_score, lpbs_target_score, disco_score, seat_score, plotbase, plotname)


def plot_results(range_list, lpbs_score, lpbs_target_score, disco_score, seat_score, plotbase, plotname):
    """Plot the results of the debiasing experiment."""
    plt.figure()
    plt.plot(range_list, lpbs_score, marker='o', linestyle='-')
    plt.xlabel('Number of Dimensions')
    plt.ylabel('log probability bias score')
    plt.title('Effect of Number of Dimensions on the log probability bias score')
    plt.savefig(f'plotSAM/lpbs_{plotbase}_{plotname}.png')
    plt.show()

    plt.figure()
    plt.plot(range_list, lpbs_target_score, marker='o', linestyle='-')
    plt.xlabel('Number of Dimensions')
    plt.ylabel('log probability bias target score')
    plt.title('Effect of Number of Dimensions on the log probability bias target score')
    plt.savefig(f'plotSAM/lpbs_target_{plotbase}_{plotname}.png')
    plt.show()

    plt.figure()
    plt.plot(range_list, disco_score, marker='o', linestyle='-')
    plt.xlabel('Number of Dimensions')
    plt.ylabel('DisCo score')
    plt.title('Effect of Number of Dimensions on the DisCo score')
    plt.savefig(f'plotSAM/disco_{plotbase}_{plotname}.png')
    plt.show()

    plt.figure()
    plt.plot(range_list, seat_score, marker='o', linestyle='-')
    plt.xlabel('Number of Dimensions')
    plt.ylabel('SEAT score')
    plt.title('Effect of Number of Dimensions on the SEAT score')
    plt.savefig(f'plotSAM/seat_{plotbase}_{plotname}.png')
    plt.show()


def run_accuracy_test(model_type, dir_folder, num_components=0):
    """Run the accuracy test on the debiased model."""
    tokenizer = AutoTokenizer.from_pretrained(model_type)
    model = AutoModelForSequenceClassification.from_pretrained(model_type)
    if num_components > 0:
        debiaser = DebiasBert(tokenizer, None, num_components, 1)
        model.set_input_embeddings(debiaser.debias_full_word_embeddings(model.get_input_embeddings()))

    output_dir = './saved_model'
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    run_glue.main(True, ['--model_name_or_path', output_dir, '--task_name', "sst2",
                         '--do_train', '--do_eval', '--max_seq_length', '128',
                         '--per_device_train_batch_size', '32', '--learning_rate', '2e-5',
                         '--num_train_epochs', '3', '--output_dir', dir_folder])


if __name__ == '__main__':
    accuracy_test = False
    debias_test = False

    if debias_test:
        model_types = ["albert-base-v2", "bert-base-uncased", "albert-large-v2", "bert-large-uncased"]
        filename1 = "results_bert-base-uncased-INPUT-DEBIAS"
        attribute_template = "This is the _."
        target_template = "This is the _."
        range_list = [2, 4, 6, 8]
        plotname = "center"
        plotbase = "bert-base"

        run_debiasing_experiment(range_list, filename1, attribute_template, target_template, plotbase, plotname)

    if accuracy_test:
        model_type = "bert-base-uncased"

        dir_folder = './test_results/bert_base'
        create_directory_if_not_exists(dir_folder)

        run_accuracy_test(model_type, dir_folder, 0)
