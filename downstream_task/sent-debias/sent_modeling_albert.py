import torch
from torch import nn
from transformers import AlbertModel, AlbertPreTrainedModel, AlbertTokenizer, AlbertConfig
from transformers.modeling_outputs import SequenceClassifierOutput

class AlbertForSequenceClassification(AlbertPreTrainedModel):
    """ALBERT model for classification with debiasing option.
    
    Params:
        config: An instance of AlbertConfig with the configuration to build a new model.
        num_labels: The number of classes for the classifier. Default is 2.
        normalize: Whether to normalize the embeddings. Default is False.
        tune_albert: Whether to fine-tune ALBERT. Default is True.
        
    Inputs:
        input_ids: A torch.LongTensor of shape [batch_size, sequence_length] with word token indices in the vocabulary.
        token_type_ids: An optional torch.LongTensor of shape [batch_size, sequence_length] with token types indices selected in [0, 1].
        attention_mask: An optional torch.LongTensor of shape [batch_size, sequence_length] with indices selected in [0, 1].
        labels: Labels for the classification output: torch.LongTensor of shape [batch_size] with indices selected in [0, ..., num_labels].
        
    Outputs:
        If `labels` is not None: CrossEntropy classification loss of the output with the labels.
        If `labels` is None: Classification logits of shape [batch_size, num_labels].
    """
    
    def __init__(self, config, num_labels=2, normalize=False, tune_albert=True):
        super().__init__(config)
        self.num_labels = num_labels
        self.albert = AlbertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.init_weights()
        self.normalize = normalize
        self.tune_albert = tune_albert
        
    def drop_bias(self, u, v):
        return u - torch.ger(torch.matmul(u, v), v) / v.dot(v)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                encode_only=False, word_level=False, remove_bias=False, bias_dir=None):
        outputs = self.albert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        pooled_output = outputs[1]

        if word_level:
            embeddings = sequence_output
            seq_length = embeddings.shape[1]
            if self.normalize:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
            if remove_bias and bias_dir is not None:
                for t in range(seq_length):
                    embeddings[:, t] = self.drop_bias(embeddings[:, t], bias_dir)
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
        else:
            embeddings = pooled_output
            if self.normalize:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
            if remove_bias and bias_dir is not None:
                embeddings = self.drop_bias(embeddings, bias_dir)
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)

        if not self.tune_albert:
            embeddings = embeddings.detach()
        if encode_only:
            return embeddings

        embeddings = self.dropout(embeddings)
        logits = self.classifier(embeddings)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return {"loss":loss, "logits":logits}
        else:
            return {"logits":logits}
