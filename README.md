# RegaVAE
This is the official repo for our [paper](https://arxiv.org/abs/2310.10567): 
> RegaVAE: A Retrieval-Augmented Gaussian Mixture Variational Auto-Encoder for Language Modeling

## Model Architecture
![](https://github.com/TrustedLLM/RegaVAE/blob/main/architecture.png)
Architecture of RegaVAE. Based on the training data, we first train a VAE to construct a compact latent space, which ensures that the latent variable z contains both current and future information (see § 3.1 of the paper). We then build a retrieval database and then aggregate the retrieved information into the generator (see § 3.2 of the paper). VAE Encoder and Decoder parameters are the same in all steps. In order to ensure fairness, the Corpus data and the Source data in the training set are the same. $G$ represents the Gaussian mixture distribution, and $π$ is the corresponding parameter.

## Datasets
Download three dataset from this [link](https://drive.google.com/drive/folders/1mcn6nqLDVvrGatKHbdbtDSj9PQI5Eu8S?usp=sharing). Unzip them and put them under the data directory.

## Step1
Firstly,
```
cd Step1
```
### Training
For Yelp dataset,
```
python main.py --train_file ../data/yelp/yelp.train.txt \
--valid_file ../data/yelp/yelp.valid.txt \
--per_gpu_train_batch_size 4 \
--cycle_annealing
```
For yahoo dataset,
```
python main.py --train_file ../data/yahoo/yahoo.train.txt \
--valid_file ../data/yahoo/yahoo.valid.txt \
--per_gpu_train_batch_size 4 \
--cycle_annealing
```
For WP dataset,
```
python main.py --train_source_path ../data/writingPrompts/train.wp_source \
--train_target_path ../data/writingPrompts/train.wp_target \
--valid_source_path ../data/writingPrompts/valid.wp_source \
--valid_target_path ../data/writingPrompts/valid.wp_target \
--dataset_type wp \
--per_gpu_train_batch_size 4 \
--cycle_annealing
```
The above are only the best adjusted hyperparameters. You can get a better Step1 model by passing other parameters. The model we trained is available at this [link](https://drive.google.com/drive/folders/1HmTqQmHSmP_VZUDV9ADM6QEHwE3SazDi?usp=sharing).
