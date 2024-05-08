# Strong Evaluation for Biometric Anonymization
Making a strong evaluating biometric anonymizations easy and reproducible.

SEBA is a framework for evaluating anonymizations for biometric data.
It is based on the evaluation methodology developed in the papers "A False Sense of Privacy: Towards a Reliable Evaluation Methodology for the Anonymization of Biometric Data" and "Fantômas: Understanding Face Anonymization Reversibility" and expands them into a joint framework which allows for easier and better comparable evaluation of anonymization techniques.
It offers an interface for both the privacy and utility evaluation of anonymizations and is easy to extend.

## Citing
When you use this framework in your paper, please cite us:
```
@article{todt_fantomas_2024,
	title = {Fantômas: Understanding Face Anonymization Reversibility},
	journal = {Proceedings on Privacy Enhancing Technologies},
	author = {Todt, Julian and Hanisch, Simon and Strufe, Thorsten},
	year = {2024},
}
```

## Fantômas
In Fantômas, we combine a range of anonymizations and de-anonymizations and run these experiments on the CelebA and DigiFace data sets.
This may be replicated by first adding these datasets and running the appropriate pre-processing (see `scripts/dataset/` and `scripts/face/`).
Then, the experiments can be run by setting the configurations from our paper up in a config file and executing them.

## A False Sense of Privacy
For *A False Sense of Privacy*, a range of anonymizations and selectors is used and experiments are run on the CelebA and WebFace260M face image data sets, as well as for gait.

## Getting started
To start using the framework, use the following guide.
Note that the framework is designed to be extendable and customizable, therefore you may not require all steps for your use-case.

1. *Clone* this repository.
```bash
git clone https://github.com/kit-ps/seba.git
```

2. *Install the base requirements.*
We highly recommend using a conda environment with python 3.10.
To get started you can use the following command, assuming you already have conda installed and configured.
```bash
conda env create -n seba -f env.yaml
```
Afterwards, you'll want to change into the just created environment.
```bash
conda activate seba
```

3. *Apply patches.*
For some of the packages you just installed, we have made slight modifications that fix bugs or change functionality to work better in our use-case.
You can apply them using the following command.
Note that our script assumes you've installed the packages in an conda environment called `seba` - as the previous command does.
```bash
scripts/apply_patches.sh
```

4. *Install optional dependencies.*
Some anonymizations, de-anonymizations, recognitions and utilities require additional dependencies which we don't bundle in the base environment file since most users will not require them.
Whenever using a new method, check its documentation to learn more.
For many, we also provide installation scripts in `scripts/installer/` that will automatically handle all requirements.

5. *Add a dataset.*
Running the framework requires datasets which we cannot package with the framework for license reasons.
Acquire and download the dataset that you want to use and place it in a folder in the `data/` directory.
Then you can use one of the scripts that we provide under `scripts/dataset/structure_` to create the structure that is required by our framework.
You may want to use some of the scripts that add attributes to the dataset or pre-process the data.

6. *Create a configuration file.*
Configuration files define what scenario and which parameters the framework will use to run an experiment.
We provide a sample as `sample.config.yaml`. Copy and modify as required. Check the documentation for details on all parameters.

7. *Run it.*
```bash
python main.py your-config-file.yaml
```


