# Exploring Calibration in the Decoding Space of Large Language Models  
**A Case Study of Machine Translation**  
by: Gerrit Janssen

This work investigates the correlation between the translation quality and model confidence in large language models (LLMs) for machine translation tasks, commonly referred to as model calibration. Specifically, it investigates how model size and supervised fine-tuning affect the degree of calibration. We first measure the correlation across differently sized models from the BLOOMZ series, which is an open-source family of models with a wide range of model sizes all trained on the same data. Our findings show that translation quality scales with size, model confidence adheres to the same trend, and consequently, calibration tends to scale as well. We then examine the influence that supervised fine-tuning has on calibration. Our experiment clearly shows that: 1) a small amount of high-quality data increases translation quality and the degree of calibration. Considering the limited amount of data, we argue that the enhanced calibration has a significant impact on the increase in translation quality from fine-tuning rather than just the addition of new knowledge. 2) Extremely high-quality data, measured by COMET, does not show clear advantages over relatively lower-quality data for fine-tuning. Similar results for both translation quality and calibration are observed for three groups of data ranging from 0.82 to 0.96. This work establishes a strong connection between translation quality and model calibration in machine translation, which future work could explore in other settings.

## Settings / Useful Installs

- Python version: `3.12.7`
- `datasets`
- `transformers`
- `bitsandbytes`
- `pytorch`
- `tensorboard`
- `peft`
  ## Important!!!
  - `in the files used for the plots in the thesis base paths are used often, change them to fit your environment`

## project structure

<pre> thesis/ # Root folder
├── exp1/ # Notebooks, scripts, and results of the first experiment
│ ├── model_{model_size}.py # All the models used in this experiment
│ ├── exp1_{model_size}.ipynb # All the notebooks used in this experiment
│ ├── csv_results_{model_size}/ # The results are stored in CSVs in folders with this naming
│ └── exp1_results/ # The plots used in the thesis are in this folder
├── exp2/ # All the models, scripts, and results of the second experiment
│ ├── finetuned_lora_model_{language_pair}{data_quality}/ # The different fine-tuned models
│ ├── translations/
│ │ ├── {language_pair}{data_quality}_gpt_model.py # The models used in the second experiment
│ │ ├── {language_pair}finetuned{data_quality}.ipynb # The notebooks used in this experiment
│ │ └── csv_results/ # The folder containing results of the experiment
│ ├── exp2_results/ # The plots used in the thesis relevant to this experiment
│ └── dataset_gpt/ # The datasets of varying quality used in this experiment</pre>



If you would like to use this work, please cite: <br>
@mastersthesis{janssen2025calibration, <br>
  title={Exploring Calibration in the Decoding Space of Large Language Models: A Case Study of Machine Translation}, <br>
  author={Janssen, Gerrit}, <br>
  school={Your University Name}, <br>
  year={2025} <br>
}
