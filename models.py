# https://medium.com/@shahrukhx01/multi-task-learning-with-transformers-part-1-multi-prediction-heads-b7001cf014bf
# https://discuss.huggingface.co/t/fine-tuning-bert-with-multiple-classification-heads/24453/8
from transformers import Wav2Vec2Model, HubertConfig, PretrainedConfig, AutoFeatureExtractor, AutoModel

class HuBERTMultiHead(Wav2Vec2Model):
    def __init__(self, config, **kwargs):
        super().__init__(PretrainedConfig())
        
        self.num_labels = kwargs.get('task_labels_map', {})
        self.config = config

        self.model_id = 'ntu-spml/distilhubert'
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            self.model_id, do_normalize=True, return_attention_mask=True
        )

        sampling_rate = self.feature_extractor.sampling_rate
        self.hubert = AutoModel.from_pretrained(self.model_id)
    # end init
# end class