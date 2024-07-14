# https://medium.com/@shahrukhx01/multi-task-learning-with-transformers-part-1-multi-prediction-heads-b7001cf014bf
# https://discuss.huggingface.co/t/fine-tuning-bert-with-multiple-classification-heads/24453/8
from transformers import Wav2Vec2Model, Wav2Vec2Config, HubertConfig, PretrainedConfig, AutoFeatureExtractor, AutoModel
from transformers.modeling_outputs import SequenceClassifierOutput
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import numpy as np

class HuBERTMultiHead(Wav2Vec2Model):
    def __init__(self,  **kwargs):
        super().__init__(Wav2Vec2Config())
        self.task_labels = kwargs.get('task_labels_map', {})
        # self.num_tasks = len( self.task_labels ) # TODO: we don't need it
        # self.config = config
        self.projector_dim = 64

        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.base_model = Wav2Vec2Model.from_pretrained("facebook/hubert-base-ls960", torch_dtype=torch.float16).to(self.dev)

        # self.model_id = 'ntu-spml/distilhubert'
        # self.feature_extractor = AutoFeatureExtractor.from_pretrained(
        #     self.model_id, do_normalize=True, return_attention_mask=True
        # )
        # self.sampling_rate = self.feature_extractor.sampling_rate
        # self.hubert = AutoModel.from_pretrained(self.model_id)

        ## add task specific output heads
        self.projectors = {}
        self.classifiers = {}
        for tl in self.task_labels.keys():
            self.projectors[tl] = nn.Linear(
                self.base_model.config.hidden_size, self.projector_dim
            )
            self.classifiers[tl] = nn.Linear(
                self.projector_dim, self.task_labels[tl]
            )
    # end init
    
    def forward(
            self,
        input_audios=None,
        attention_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        task_name=None,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        audio_normalized = self.feature_extractor(
            input_audios,
            sampling_rate=self.sampling_rate,
            return_attention_mask=True,
        )
        
        audio_tensors = torch.from_numpy(np.array(audio_normalized['input_values']))
        attention_mask = torch.from_numpy(np.array(audio_normalized['attention_mask']))
        
        outputs = self.hubert(
            audio_tensors,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        pooled_y = outputs['last_hidden_state'].mean(axis=1)

        y = self.projectors[task_name]
        logits = self.classifiers[task_name]

        self.problem_type = None
        loss = None
        # if labels are given, i.e., if in training mode
        if labels is not None:
            # check the type of problem, i.e., regression or singe/multi classfication
            if self.problem_type is None:
                if self.num_labels[task_name] == 1:
                    self.problem_type = "regression"
                elif self.num_labels[task_name] > 1 and (
                    labels.dtype == torch.long or labels.dtype == torch.int
                ):
                    self.problem_type = "single_label_classification"
                else:
                    self.problem_type = "multi_label_classification"
        # apply loss
            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels[task_name] == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(
                    logits.view(-1, self.num_labels[task_name]), labels.view(-1)
                )
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
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
# end class