# https://medium.com/@shahrukhx01/multi-task-learning-with-transformers-part-1-multi-prediction-heads-b7001cf014bf
# https://discuss.huggingface.co/t/fine-tuning-bert-with-multiple-classification-heads/24453/8
from transformers import Wav2Vec2Model, Wav2Vec2Config, Wav2Vec2FeatureExtractor
from transformers.modeling_outputs import SequenceClassifierOutput
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import numpy as np

class HuBERTMultiHead(Wav2Vec2Model):
    def __init__(self,  **kwargs):
        super().__init__(Wav2Vec2Config())
        self.num_labels = kwargs.get('task_labels_num_out', {})
        self.task_labels = list(self.num_labels.keys())
        # self.num_tasks = len( self.task_labels ) # TODO: we don't need it
        # self.config = config
        self.projector_dim = 64

        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.hubert = Wav2Vec2Model.from_pretrained("facebook/hubert-base-ls960", torch_dtype=torch.float16).to(self.dev)
        self.audio_normalizer = Wav2Vec2FeatureExtractor.from_pretrained(
            "facebook/hubert-base-ls960"
        )
        self.sampling_rate = self.audio_normalizer.sampling_rate

        ## add task specific output heads
        self.projectors = {}
        self.classifiers = {}
        for tl in self.task_labels:
            self.projectors[tl] = nn.Linear(
                self.hubert.config.hidden_size, self.projector_dim
            ).half().to(self.dev)
            self.classifiers[tl] = nn.Linear(
                self.projector_dim, self.num_labels[tl]
            ).half().to(self.dev)
    # end init
    
    def forward(
        self,
        audio_normalized=None,
        attention_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None
    ):
        
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        
        audio_tensors = torch.from_numpy(np.array(audio_normalized)).half().to(self.dev)
        attention_mask = torch.from_numpy(np.array(audio_normalized)).half().to(self.dev)

        outputs = self.hubert(
            audio_tensors,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        pooled_y = outputs['last_hidden_state'].mean(axis=1)
        logits = None
        loss = None
        self.problem_type = None

        for task_name in self.task_labels:
            y = self.projectors[task_name](pooled_y)
            logits = self.classifiers[task_name](y)
            self.problem_type = None
            
            # if labels are given, i.e., if in training mode
            if labels is not None:
                # check the type of problem, i.e., regression or singe/multi classfication
                if self.problem_type is None:
                    if self.num_labels[task_name] == 1:
                        self.problem_type = "regression"
                    elif self.num_labels[task_name] > 1 and (
                        isinstance(labels[task_name][0], int)
                    ):
                        print('single_label_classification')
                        self.problem_type = "single_label_classification"
                    else:
                        print('multi_label_classification')
                        self.problem_type = "multi_label_classification"
                # apply loss
                if self.problem_type == "regression":
                    loss_fn = MSELoss()
                    print('regression 0')
                    if self.num_labels[task_name] == 1:
                        if loss is None:
                            print('regression 1')
                            # loss = loss_fn(logits.squeeze(), labels[task_name].squeeze())
                            loss = loss_fn(logits.squeeze(), torch.FloatTensor(labels[task_name]).to(self.dev))
                        else:
                            print('regression 2')
                            # loss += loss_fn(logits.squeeze(), labels[task_name].squeeze())
                            loss += loss_fn(logits.squeeze(), torch.FloatTensor(labels[task_name]).to(self.dev))
                    else:
                        print('regression 3')
                        loss = loss_fn(logits, labels)
                elif self.problem_type == "single_label_classification":
                    loss_fn = CrossEntropyLoss()
                    print('single_label_classification 0')
                    if loss is None:
                        print('single_label_classification 1')
                        loss = loss_fn(
                            logits.view(-1, self.num_labels[task_name]), labels.view(-1)
                        )
                    else:
                        print('single_label_classification 2')
                        loss += loss_fn(
                            logits.view(-1, self.num_labels[task_name]), labels.view(-1)
                        )
                elif self.problem_type == "multi_label_classification":
                    loss_fn = BCEWithLogitsLoss()
                    print('multi_label_classification 0')
                    if loss is None:
                        print('multi_label_classification 1')
                        loss = loss_fn(logits, labels)
                    else:
                        print('multi_label_classification 2')
                        loss += loss_fn(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs['last_hidden_state'],
            attentions=outputs.attentions
        )
    # end forward

    def collate_fn(self, data):
        input_values = [d['input_values'] for d in data]
        # labels needs to be a dict of lists
        labels = {}
        for d in data:
            for k in d['labels'].keys():
                labels.setdefault(k,[]).append( d['labels'][k] )
        # labels = [d['labels'] for d in data]
        audio_normalized = self.audio_normalizer(
            input_values,
            sampling_rate=self.sampling_rate,
            padding=True,
            return_attention_mask=True,
        )
        return audio_normalized, labels
    # end collate_fn
# end class