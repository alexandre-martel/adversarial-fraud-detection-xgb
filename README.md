# Adversarial and Privacy Reinforced Credit Card Fraud Detection on MLP

A credit card fraud detection project using MLP that implements defenses against three issues: adversarial attacks (FGSM), privacy attacks (DG-SDG), and bias mitigation.
The datset used for this project is the following : 

## Explanations and Mathematical Theory

### Adversarial Attack (FGSM) /Defense Method (Adversarial training)

#### Adversarial Attack : FGSM
#### Adversarial Defense : Adversarial training
#### Results

### Privacy Attack (Membership Ineference Attack) /Defense Method (DP-SGD)

#### Privacy Attack : Membership Ineference Attack
#### Privacy Defense : DP-SGD
#### Results


### Bias & Mitigation 

## Download and Run

### Downloads

Download the Github

```bash
git clone https://github.com/alexandre-martel/adversarial-privacy-fraud-detection.git
cd adversarial-privacy-fraud-detection
```

Doawnload the dataset Credit Card Fraud Detection

```bash
python -m src.utils -d
```

### Run baselines 

Run training of the baseline MLP model and summarize on val/test data.

```bash
python -m src.baselines.baseline_mlp
```

Arguments : 

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--data-path` | Path for the data in data/filename.csv | `data/creditcard.csv` |
| `--seed` | Seed for randomness | `9` |
| `--batch-size` | Batch size of the training  | `512` |
| `--epochs` | Number of epochs  | `10` |
| `--lr` | Learning rate of the training  | `1e-3` |


### Run Adversarial Attack

Run an adversarial attack on a given model and summarize on val/test data.

```bash
python -m src.adversarial.fsgm_attack
```

Arguments : 

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--epsilon` | Perturbation strength for FGSM attack | `0.1` |
| `--batch-size` | Batch size for generating adversarial examples | `512` |
| `--model-folder` | Folder where the baseline model and preprocessing objects are saved | `baseline_model` |

### Run Adversarial Training + Adversarial Attack

Run an adversarial traning with the base model and then an attack on a that specific trained model and summarize on test data for both in order to compare.

```bash
python -m src.adversarial.aversarial_training
```

Arguments : 

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--data-path` | Path for the data in data/filename.csv | `data/creditcard.csv` |
| `--seed` | Seed for randomness | `9` |
| `--batch-size` | Batch size of the training  | `512` |
| `--epochs` | Number of epochs  | `10` |
| `--lr` | Learning rate of the training  | `1e-3` |
| `--epsilon` | Perturbation strength for FGSM attack | `0.1` |
| `--mix-ratio` | Ratio of adversarial samples in the mixed training batches | `0.5` |
