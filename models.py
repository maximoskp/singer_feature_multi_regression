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

        self.hubert = Wav2Vec2Model.from_pretrained("facebook/hubert-base-ls960", torch_dtype=torch.float32).to(self.dev)
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
            ).to(self.dev)
            self.classifiers[tl] = nn.Linear(
                self.projector_dim, self.num_labels[tl]
            ).to(self.dev)
            # self.projectors[tl] = nn.Linear(
            #     self.hubert.config.hidden_size, self.projector_dim
            # ).half().to(self.dev)
            # self.classifiers[tl] = nn.Linear(
            #     self.projector_dim, self.num_labels[tl]
            # ).half().to(self.dev)
        self.relu = nn.LeakyReLU()
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
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
        
        # audio_tensors = torch.from_numpy(np.array(audio_normalized)).half().to(self.dev)
        # attention_mask = torch.from_numpy(np.array(audio_normalized)).half().to(self.dev)
        audio_tensors = torch.from_numpy(np.array(audio_normalized)).to(self.dev)
        attention_mask = torch.from_numpy(np.array(audio_normalized)).to(self.dev)

        outputs = self.hubert(
            audio_tensors,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        pooled_y = outputs['last_hidden_state'].sum(axis=1)
        # pooled_y = outputs['last_hidden_state'][:,-1,:]
        logits = None
        loss = None
        self.problem_type = None
        task_logits = {}
        pooled_y = self.relu( pooled_y )
        print('pooled: ', pooled_y)

        for task_name in self.task_labels:
            y = None
            logits = None
            y = self.projectors[task_name](pooled_y)
            print('projector y: ', y)
            y = self.relu( y )
            print('projector relu y: ', y)
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
                        self.problem_type = "single_label_classification"
                    else:
                        self.problem_type = "multi_label_classification"
                # apply loss
                if self.problem_type == "regression":
                    y = self.classifiers[task_name](y)
                    print('classifiers y: ', y)
                    logits = self.sigmoid(y)
                    print('sigmoid logits: ', logits)
                    loss_fn = MSELoss()
                    if self.num_labels[task_name] == 1:
                        if loss is None:
                            # loss = loss_fn(logits.squeeze(), labels[task_name].squeeze())
                            # loss = loss_fn(logits.squeeze().half(), torch.FloatTensor(labels[task_name]).half().to(self.dev))
                            # loss = loss_fn(logits.half(), torch.FloatTensor(labels[task_name]).half().to(self.dev))
                            loss = loss_fn(logits, torch.FloatTensor(labels[task_name]).to(self.dev))
                            print('sigmoid logits 1: ', logits)
                        else:
                            # loss += loss_fn(logits.squeeze(), labels[task_name].squeeze())
                            # loss += loss_fn(logits.squeeze().half(), torch.FloatTensor(labels[task_name]).half().to(self.dev))
                            # loss += loss_fn(logits.half(), torch.FloatTensor(labels[task_name]).half().to(self.dev))
                            loss = loss_fn(logits, torch.FloatTensor(labels[task_name]).to(self.dev))
                            print('sigmoid logits 2: ', logits)
                    else:
                        loss = loss_fn(logits, labels)
                        print('sigmoid logits 3: ', logits)
                    print('task_name: ', task_name)
                    # print(torch.FloatTensor(labels[task_name]).half().to(self.dev))
                    print(torch.FloatTensor(labels[task_name]).to(self.dev))
                    print('logits squeeze', logits)
                    print('loss 0: ', loss)
                elif self.problem_type == "single_label_classification":
                    y = self.classifiers[task_name](y)
                    logits = self.softmax(y)
                    loss_fn = CrossEntropyLoss()
                    if loss is None:
                        loss = loss_fn(
                            logits.view(-1, self.num_labels[task_name]), torch.LongTensor(labels[task_name]).to(self.dev).view(-1)
                        )
                    else:
                        loss += loss_fn(
                            logits.view(-1, self.num_labels[task_name]), torch.LongTensor(labels[task_name]).to(self.dev).view(-1)
                        )
                    print('loss 1: ', loss)
                elif self.problem_type == "multi_label_classification":
                    y = self.classifiers[task_name](y)
                    logits = self.sigmoid(y)
                    loss_fn = BCEWithLogitsLoss()
                    if loss is None:
                        # loss = loss_fn(logits, torch.FloatTensor(labels[task_name]).half().to(self.dev))
                        loss = loss_fn(logits, torch.FloatTensor(labels[task_name]).to(self.dev))
                    else:
                        # loss += loss_fn(logits, torch.FloatTensor(labels[task_name]).half().to(self.dev))
                        loss += loss_fn(logits, torch.FloatTensor(labels[task_name]).to(self.dev))
                    print('loss 2: ', loss)
                task_logits.setdefault(task_name, []).append(logits)
                break
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        return SequenceClassifierOutput(
            loss=loss,
            logits=task_logits,
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