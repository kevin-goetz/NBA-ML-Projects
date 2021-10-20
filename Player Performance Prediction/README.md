# NBA Rookie Career PER Prediction

## Access

### Online

View the notebooks online:
[![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/kevin-goetz/NBA-ML-Projects/tree/main/Player%20Performance%20Prediction/Notebooks/?flush_cache=true)

Excecute the notebooks online: 
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/kevin-goetz/NBA-ML-Projects/HEAD)

This can take some time if the binder environment needs to be rebuilt (navigate from the parent folder to the projects notebooks).
 
### Offline

For the Notebooks two different virtual environments (venv) had to be set up because PyCaret only runs with a different version of Python and scikit-learn (older ones) compared to the newer versions that were used for the rest of the project. If you don't want to run the "Modelling with Pycaret" - Notebook it's sufficient to work with only one venv and install the **requirements.txt** file.

## Description

### What
This project tried to answer the question: "What is the Career Performance of a Rookie going to be like?". The predictors were player performance metrics from his rookie season, as well as data from the Draft (college, nba team, pick, etc.). The target was the Player Efficiency Rating (PER) averaged over the whole career.

### How
By collecting data from a [Database (Kaggle)](https://www.kaggle.com/wyattowalsh/basketball) with SQL and enriching this data with further performance metrics by web scraping the website ["basketball-reference.com"](https://www.basketball-reference.com) for each individual player on all the years the player was active in the NBA.

The data was then preprocessed and condensed with a PCA (Principal Component Analysis) and an RFE (Recurive Feature Elimination) before being modelled with a VotingRegressor (GradientBoostingRegressor & RandomForestRegressor) all in a ML Pipeline (see it below!) to avoid data leakage.

### Why
For NBA teams and their managers it is very important to foresee the future performance and the potential of a player at an early stage. the project is intended to help support the experts' qualitative assessment with a quantitative analysis and prediction.

### Results & Impact
With an R2 of 0.44 the results are not too impactful, there is still a lot to learn about the factors of a successful career (discussion in "Outlook" below) and also how to model the data in an optimal way. The model overestimates untalented rookies and also understimates talented rookies regarding their career PER. The following error plot summarizes this:

<img src="https://github.com/kevin-goetz/NBA-ML-Projects/blob/main/Player%20Performance%20Prediction/Models/Prediction%20Error.PNG" align="center" height="300em" />

The table of the Top 5 rookies is to be taken with a grain of salt therefore:

| namePlayer| numberPickOverall | Age | Tm | Pos | Rookie_PER | Predicted_Career_PER |
|---:|---:|---:|---:|---:|---:|---|
| LaMelo Ball | 3 | 19.0 | CHO | PG | 17.5 | 19.48 |
| Isaiah Stewart | 16 | 19.0 | DET | C | 16.4 | 17.07 |
| Onyeka Okongwu | 6 | 20.0 | ATL | C | 16.8 | 16.69 |
| Tyrese Haliburton | 12 | 20.0 | SAC | PG | 16.2 | 16.32 |
| Kenyon Martin Jr. | 52 | 20.0 | HOU | SF | 14.6 | 16.07 |

## The Machine Learning Pipeline

Here's a HTML representation of the ML pipeline to get an overview of the process of preparing and modelling the data.

**--> Click on it to see the interactive HTML for details** 👇

[![raw.githack.com](https://github.com/kevin-goetz/NBA-ML-Projects/blob/main/Player%20Performance%20Prediction/Models/ML%20Pipeline.PNG)](https://rawcdn.githack.com/kevin-goetz/NBA-ML-Projects/9c439f9f4304749febc12af72782517efaa1d8ee/Player%20Performance%20Prediction/Models/NBA_Rookie_model_pipeline.html)


## Skills
Technical skills honed in this project are:
- Database Integration in a Python Script (SQL)
- Web Scraping (Requests, BeautifulSoup)
- Data Wrangling (Pandas, NumPy)
- Data Visualization & EDA (pandas-profiling, Matplotlib, Seaborn, missingno, plotly, Yellowbrick)
- Outlier Detection with Z-Scores, IQR and Isolation Forest (Pandas, NumPy, scikit-learn, Matplotlib, Plotly)
- ML Pipelines (scikit-learn, PyCaret)
- Encoding, Imputation & Scaling / Transforming (scikit-learn)
- Feature Selection (RFE) & Dimensionality Reduction (PCA) (scikit-learn)
- Hyperparametertuning (scikit-learn)
- Ensemble Meta-Estimators (scikit-learn)

## Personal Learnings:

**1. Web Scraping is tricky**:<br/>
Dealing with unstructured data from websites is always a hard job, especially when scraping thousands of different sites based on the structure of the URL. The most tricky      part is not getting the data, but deciding if the scraped data is indeed the correct data (or did the program scrape data from another players profile because they have the same name?). Also dealing with bot blocking or rate limits can make the process hard, but not impossible. A special kind of challenge is ne never ending encoding, representation, and handling of text with special characters (ASCII vs. Unicode etc.). I found the library "unidecode" to be easy and useful for those tasks.

**2. ML Pipelines are a great thing**:<br/>
ML Pipelines (meaning ColumnTransfer and FeatureUnion as well) make the life of a Data Scientist much easier:
- They enforce you to think about the order of steps in your project.
- They make you work in a more structured and repeatable way.
- They make your workflow much easier to read and understand.
- These in turn make your work much more reproducible.
- They help in preventing data leakage (which is probably the most important).

Pipelines are one honking great idea -- let's do more of those!


**3. Low-Code ML Libraries are huge time-savers**: <br/>
Sometimes the ultimate goal isn't the uttermost R2 or Accuracy score, but to provide as much added value as possible in a short time. When timings are tight and deadlines are approaching fast, so called low-code ml or auto-ml libraries come in handy. For this project I tried [PyCaret](https://pycaret.org/), which is an open source, low-code machine learning library in Python that allows you to go from preparing your data to deploying your model within no time in your choice of notebook environment. I benchmarked the PyCaret model against the scikit-learn model and indeed, the results are quiet comparable. Even though scikit-learn offers much more flexibility, PyCaret is way faster (coding-wise) and the two are not exclusive, for sure. My approach for the future will be to preprocess in sklearn, then test different algorithms in PyCaret, and then proceed in sklearn with the most promising algorithms to fine-tune them.

**4. Don't over-engineer**: <br/>
Sometimes simple solutions are efficient enough, sometimes they are even better than the complex ones. It is not only about the skill metrics of the model, but about how fast a model gets deployed and how much compute ressources it needs to run. Even if it does not get deployed because it is an ad-hoc analysis: the hyperparamtertuning can take up a lot of time if the pipeline is complex. To cite the Zen of Python: Simple is better than complex.


## Outlook
Predicting a players career is a hard task because it depends on much more than only the rookie year stats. Injuries, Lifestyle, Marketing and Teammates all contribute to the long term success of a player and weren't a variable in the prediction model. Future research could focus on those variables from different data sources as well to boost the skill of the model.

Furthermore, the data cutoff during collection (veterans >= 5 years of experience) was consciously chosen but can be extended or lessened for further tests. Also, all the selected players were drafted 2001 or later which could be extended to the 90's to get more data and probably a more stable estimator (generalization), which was a problem during model training and testing. 


## 📫 Let's connect and learn from each other:

[<img src="https://github.com/kevin-goetz/kevin-goetz/blob/main/LinkedIn Logo.png" height="40em" align="center" alt="Connect with Me on LinkedIn" title="Connect with Me on LinkedIn"/>](https://linkedin.com/in/kgötz) [<img src="https://github.com/kevin-goetz/kevin-goetz/blob/main/Codewars Logo.svg" height="40em" align="center" alt="Connect with Me on Codewars" title="Connect with Me on Codewars"/>](https://www.codewars.com/users/kevin-goetz)


