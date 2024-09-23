# singer_feature_multi_regression
Applies multiple regression heads to fine tune an audio transformer on multiple downstream regression tasks. The base for all models is a frozer HuBERT model and on top of that, different architectures are tried out.

So far the investigation concerns three architectures (but there seems to be something wrong because results don't make absolute sense):

Baseline singer classification model (no features involved):
![Alt text](images/baseline.jpg)
![Alt text](images/baseline_acc.png)

No fusion model (feature regression and singer classification are preformed independently):
![Alt text](images/no_fusion.jpg)
![Alt text](images/nofusion_acc.png)

Early fusion model (feature regression and singer classification have the same trainable base):
![Alt text](images/early_fusion.jpg)
![Alt text](images/early_acc.png)

Late fusion model (singer classification is based on the projectors of feature regression modules and training occurs jointly):
![Alt text](images/late_fusion.jpg)
![Alt text](images/late_acc.png)