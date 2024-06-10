from transformers import BertTokenizer, BertForMaskedLM,  BertModel, AlbertForMaskedLM, AlbertTokenizer
import torch
from debias_class import DebiasBert
from data.professions import get_all_professions

def predict_and_evaluate(word, model, tokenizer):
    # Input sentence
    sentence = f"The function is {word}."

    # Tokenize input
    tokens = tokenizer.tokenize(sentence)
    encoded_inputs = tokenizer.encode(sentence, return_tensors='pt')

    # Predict masked tokens
    with torch.no_grad():
        outputs = model(encoded_inputs)
        logits = outputs.logits

    # Apply softmax to logits and find the token with the highest probability
    predictions = []
    for i in range(4, len(logits[0]) - 2):
        predicted_token_index = torch.argmax(logits[0][i]).item()
        predicted_token = tokenizer.convert_ids_to_tokens([predicted_token_index])[0]
        predictions.append((predicted_token, tokens[i - 1]))

    return predictions


def evaluate_model(model, tokenizer):
    professions = get_all_professions()
    total = 0
    amountwrong = 0

    for word in professions:
        predictions = predict_and_evaluate(word, model, tokenizer)
        for predicted_token, actual_token in predictions:
            if predicted_token != actual_token:
                # print(f"Predicted: {predicted_token}, Actual: {actual_token}, Word: {word}")
                amountwrong += 1
            total += 1
    accuracy = (total - amountwrong) / total if total > 0 else 0

    return accuracy, total, amountwrong


if __name__ == '__main__':
    model_type1 = "albert-base-v2"
    tokenizer = AlbertTokenizer.from_pretrained(model_type1)
    model = AlbertForMaskedLM.from_pretrained(model_type1)
    debiaser = DebiasBert(tokenizer, None, 10, 1)
    old_emb = model.get_input_embeddings()
    emb = debiaser.debias_full_word_embeddings(old_emb)
    model.set_input_embeddings(emb)

    model.eval()

    accuracy, total, amountwrong = evaluate_model(model, tokenizer)

    print(f"Total: {total}")
    print(f"Total wrong: {amountwrong}")
    print(f"Accuracy: {accuracy:.2%}")