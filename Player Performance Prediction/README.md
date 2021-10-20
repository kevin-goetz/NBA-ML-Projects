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
Project on predicting (regression) a NBA rookie's career PER (Player Efficiency Rating averaged over his whole career) by just looking at his first season stats.

### How
By collecting data from a [Database (Kaggle)](https://www.kaggle.com/wyattowalsh/basketball) with SQL and enriching this data with further performance metrics by web scraping the website ["basketball-reference.com"](https://www.basketball-reference.com) for each individual player on all the years the player was active in the NBA.

The data was then preprocessed and condensed with a PCA (Principal Component Analysis) and an RFE (Recurive Feature Elimination) before being modelled with a VotingRegressor (GradientBoostingRegressor & RandomForestRegressor) all in a ML Pipeline (see it below!) to avoid data leakage.

### Why
For NBA teams and their managers it is very important to foresee the future performance and the potential of a player at an early stage. the project is intended to help support the experts' qualitative assessment with a quantitative analysis and prediction.

### Results & Impact
....

## The Machine Learning Pipeline

Here's a HTML representation of the ML pipeline to get an overview of the process of preparing and modelling the data.

  **--> Click on it to get the interactive HTML for details!**

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
....

## Outlook
... more data bla bla


## 📫 Let's connect and learn from each other:

[<img src="https://github.com/kevin-goetz/kevin-goetz/blob/main/LinkedIn Logo.png" height="40em" align="center" alt="Connect with Me on LinkedIn" title="Connect with Me on LinkedIn"/>](https://linkedin.com/in/kgötz) [<img src="https://github.com/kevin-goetz/kevin-goetz/blob/main/Codewars Logo.svg" height="40em" align="center" alt="Connect with Me on Codewars" title="Connect with Me on Codewars"/>](https://www.codewars.com/users/kevin-goetz)


