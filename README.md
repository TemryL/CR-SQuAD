# Context Retrieval on SQuAD Dataset

* [Getting started](#getting-started)
    * [Project description](#project-description)
    * [Data](#data)
* [Reproduce results](#reproduce-results)
    * [Requirements](#Requirements)
    * [Repo Architecture](#repo-architecture)
    * [Instructions to run](#instructions-to-run)
* [Results](#results)

# Getting started
This repository contains the codes used to produce the results presented in `report.pdf`
## Project description


## Data
The raw data are already available in the data folder.

# Reproduce results
## Requirements

- python=3.10.8
- pytorch=1.13.1
- pandas=1.5.2
- scikit-learn=1.1.3
- tqdm=4.64.1
- matplotlib=3.6.2
- seaborn=0.12.1

## Repo Architecture
<pre>  
├─── bestmodels
    ├─── commodities
├─── data
├─── README.md: README
├─── report.pdf: Report explaining methods and choices that have been made.
├─── requirements.txt: requirements
├─── results.txt: Results in Table II and Table III of `reports.pdf`

</pre>

## Instructions to run 
First make sure to have all the requirements and the data folder in the root.

The following commands give more details about the positional arguments and a description of the process done while running:

```
python process_data.py -h
python validation.py -h
python train.py -h
python test.py -h
```
Please run them before running the following. The commands showed bellow have to be executed in the same order to keep consistency.

The processed data can be reproduced from the raw data by moving to the `src/` folder and execute:
```
python process_data.py dataset nb_lags train_ratio
````

To run the optimization on the validation set move to the `src/` folder and execute:
```
python validation.py model_type dataset
```
Beware that optimizing one model type on one dataset takes from 1min to 8 min (depending on the model) on Google Colab with GPU availability.

To train the models with the best parameters found during the validation move to the `src/` folder and execute:
```
python train.py model_type dataset
```

To test the performances of the trained models move to the `src/` folder and execute:
```
python test.py model_type dataset
````
# Results

### Hit-Rate comparison for each asset
|          | B&H |  NN                     | CNN                    | LSTM  | RF | C1            | C2
|-------------------|------------|-------|---------------------------------|---------------------------------|----------------|-------------|------------------------|
| Bitcoin  | 0.473      | 0.457 | 0.446                           | 0.479                  | 0.459          | 0.451       | `0.505`
| Ethereum | 0.490      | 0.457 | 0.490                  | 0.481                           | 0.475          | 0.488       | `0.497`
| Ripple   | 0.490      | 0.497 | 0.497                           | `0.532` | 0.470          | 0.514       | 0.525
| Nat. gas | 0.516      | 0.518 | `0.519` | 0.494                           | 0.498          | 0.505       | 0.502
| Gold     | 0.493      | 0.502 | 0.511                           | 0.477                           | 0.512 | 0.509       | `0.514`
| Oil      | 0.562      | 0.492 | 0.508                           | 0.527                  | 0.522          | 0.525       | `0.536`
| SP&500    |  0.564 | 0.553                           | 0.554                           | 0.558 | 0.548       | `0.572` | 0.533
| CAC40    | 0.548      | 0.520 | 0.526                           | `0.528` | 0.498          | 0.504       | 0.520
| SMI      | 0.535      | 0.507 | 0.511                  | 0.506                           | 0.488          | 0.517       | `0.546`

### Sharpe ratio comparison for each asset
|          | B&H | NN                     | CNN                    | LSTM  | RF | C1           | C2 |
|-------------------|------------|--------|---------------------------------|---------------------------------|----------------|-------------|------------------------|
| Bitcoin  | -1.163     | -0.981 | -2.066                          | -0.140                | -2.020         | -1.856      | `0.413` |
| Ethereum | -0.870     | -0.560 | -0.327                          | `1.123` | 0.245          | 0.308       | -0.404                 |
| Ripple   | -0.947     | -0.143 | 0.521                           | 1.191                  | -1.453         | 0.531       | `1.497` |
| Nat. gas | 0.743      | 0.523  | `1.107` | -0.308                          | 0.352          | 0.577       | 0.615                  |
| Gold     | 0.034      | -0.056 | -0.960                          | -0.468                          | 0.305 | -0.390      | `0.596` |
| Oil      | 0.686      | 0.389  | 0.652                           | 0.994                           | 1.012 | 0.810       | `1.049` |
| S&P500      | 0.518  | 0.833                  | 0.801                           | 0.768          | 0.090       | `1.136` | 0.551       |
| CAC40    | 0.597      | -0.192 | `0.702` | 0.499                           | -0.674         | 0.169       | 0.685                  |
| SMI      | 0.219      | -0.588 | -0.321                 | -0.367                          | -0.805         | -0.400      | `0.522` |

<pre>
</pre>
