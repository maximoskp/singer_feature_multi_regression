# https://medium.com/@shahrukhx01/multi-task-learning-with-transformers-part-1-multi-prediction-heads-b7001cf014bf
# https://discuss.huggingface.co/t/fine-tuning-bert-with-multiple-classification-heads/24453/8
from transformers import Wav2Vec2Model, Wav2Vec2Config, HubertConfig, PretrainedConfig, AutoFeatureExtractor, AutoModel
import torch
from torch import nn

class HuBERTMultiHead(Wav2Vec2Model):
    def __init__(self,  **kwargs):
        super().__init__(Wav2Vec2Config())
        
        self.num_labels = kwargs.get('task_labels_map', {})
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

        ## add task specific output heads - TODO: make a list of linear
        self.projector1 = nn.Linear(
            self.base_model.config.hidden_size, self.projector_dim
        )
        self.classifier1 = nn.Linear(
            self.projector_dim, list(self.num_labels.values())[0]
        )
        self.projector2 = nn.Linear(
            self.base_model.config.hidden_size, self.projector_dim
        )
        self.classifier2 = nn.Linear(
            self.projector_dim, list(self.num_labels.values())[1]
        )
    # end init
    
    def forward(
            self,
        input_audios=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        task_name=None,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        inputs = self.feature_extractor(
            input_audios,
            sampling_rate=self.sampling_rate,
            return_attention_mask=True,
        )

        outputs = self.hubert(
            inputs,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
    # end forward
# end class