# NBA Rookie Career PER Prediction

## Access

### Online

View the notebooks online:
[![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/kevin-goetz/NBA-ML-Projects/tree/main/Player%20Performance%20Prediction/Notebooks/)

Excecute the notebooks online: 
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/kevin-goetz/NBA-ML-Projects/HEAD)

This can take some time if the binder environment needs to be rebuilt (navigate from the parent folder to the projects notebooks).
 
### Offline

For the Notebooks two different virtual environments (venv) had to be set up because PyCaret only runs with a different version of Python and scikit-learn (older ones) compared to the newer versions that were used for the rest of the project. If you don't want to run the "Modelling with Pycaret" - Notebook it's sufficient to work with only one venv and install the **requirements.txt** file.

## Description

### What
Project on how to optimize the default pandas data types that are inferred when reading a csv-file. The result is a dataframe that takes up much less RAM without the loss of any information in the data.

### How
By making use of the pd.read_csv parameters, downcasting the numerical values, and representing columns with low cardinality in a different way (categories for strings and sparse arrays for numbers). A detailed explanation of how is listed in my personal learnings below.

### Why
Pandas provides data structures for in-memory analytics, which makes using pandas to analyze datasets that are larger than memory somewhat tricky. Even datasets that are a sizable fraction of memory become unwieldy, as some pandas operations need to make intermediate copies. With small data (under 100 megabytes), performance is rarely a problem. When we move to larger data (100 megabytes to multiple gigabytes), performance issues can make run times much longer, and cause code to fail entirely due to insufficient memory.

That's what this Notebook is all about: How to shrink your pandas dataframe so it fit's your RAM better - without losing any information.

## The Machine Learning Pipeline

Here's a HTML represeantation of the ML pipeline to get an overview of the process of preparing and modelling the data:

<style>#sk-14b9baff-6a3a-44f7-96c4-77e8a9f97aaf {color: black;background-color: white;}#sk-14b9baff-6a3a-44f7-96c4-77e8a9f97aaf pre{padding: 0;}#sk-14b9baff-6a3a-44f7-96c4-77e8a9f97aaf div.sk-toggleable {background-color: white;}#sk-14b9baff-6a3a-44f7-96c4-77e8a9f97aaf label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-14b9baff-6a3a-44f7-96c4-77e8a9f97aaf div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-14b9baff-6a3a-44f7-96c4-77e8a9f97aaf div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-14b9baff-6a3a-44f7-96c4-77e8a9f97aaf input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-14b9baff-6a3a-44f7-96c4-77e8a9f97aaf div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-14b9baff-6a3a-44f7-96c4-77e8a9f97aaf div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-14b9baff-6a3a-44f7-96c4-77e8a9f97aaf input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-14b9baff-6a3a-44f7-96c4-77e8a9f97aaf div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-14b9baff-6a3a-44f7-96c4-77e8a9f97aaf div.sk-estimator:hover {background-color: #d4ebff;}#sk-14b9baff-6a3a-44f7-96c4-77e8a9f97aaf div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-14b9baff-6a3a-44f7-96c4-77e8a9f97aaf div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-14b9baff-6a3a-44f7-96c4-77e8a9f97aaf div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 2em;bottom: 0;left: 50%;}#sk-14b9baff-6a3a-44f7-96c4-77e8a9f97aaf div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;}#sk-14b9baff-6a3a-44f7-96c4-77e8a9f97aaf div.sk-item {z-index: 1;}#sk-14b9baff-6a3a-44f7-96c4-77e8a9f97aaf div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;}#sk-14b9baff-6a3a-44f7-96c4-77e8a9f97aaf div.sk-parallel::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 2em;bottom: 0;left: 50%;}#sk-14b9baff-6a3a-44f7-96c4-77e8a9f97aaf div.sk-parallel-item {display: flex;flex-direction: column;position: relative;background-color: white;}#sk-14b9baff-6a3a-44f7-96c4-77e8a9f97aaf div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-14b9baff-6a3a-44f7-96c4-77e8a9f97aaf div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-14b9baff-6a3a-44f7-96c4-77e8a9f97aaf div.sk-parallel-item:only-child::after {width: 0;}#sk-14b9baff-6a3a-44f7-96c4-77e8a9f97aaf div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;position: relative;}#sk-14b9baff-6a3a-44f7-96c4-77e8a9f97aaf div.sk-label label {font-family: monospace;font-weight: bold;background-color: white;display: inline-block;line-height: 1.2em;}#sk-14b9baff-6a3a-44f7-96c4-77e8a9f97aaf div.sk-label-container {position: relative;z-index: 2;text-align: center;}#sk-14b9baff-6a3a-44f7-96c4-77e8a9f97aaf div.sk-container {display: inline-block;position: relative;}</style><div id="sk-14b9baff-6a3a-44f7-96c4-77e8a9f97aaf" class"sk-top-container"><div class="sk-container"><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="85c8c547-b046-4d1f-814f-e07787f4fce9" type="checkbox" ><label class="sk-toggleable__label" for="85c8c547-b046-4d1f-814f-e07787f4fce9">TransformedTargetRegressor</label><div class="sk-toggleable__content"><pre>TransformedTargetRegressor(regressor=Pipeline(steps=[('rfe_preprocessor',
                                                      Pipeline(steps=[('preprocessor',
                                                                       ColumnTransformer(n_jobs=-1,
                                                                                         transformers=[('onehot_org',
                                                                                                        Pipeline(steps=[('SI',
                                                                                                                         SimpleImputer(fill_value='missing',
                                                                                                                                       strategy='constant')),
                                                                                                                        ('OHE_Org',
                                                                                                                         OneHotEncoder(categories=[['College/University',
                                                                                                                                                    'High '
                                                                                                                                                    'School',
                                                                                                                                                    'Other '
                                                                                                                                                    'Team/Club']],
                                                                                                                                       drop='first',
                                                                                                                                       ha...
                                                                      ('rfe',
                                                                       RFE(estimator=RandomForestRegressor(),
                                                                           n_features_to_select=12))])),
                                                     ('regressor',
                                                      VotingRegressor(estimators=[('GB',
                                                                                   GradientBoostingRegressor(criterion='squared_error',
                                                                                                             max_features='auto')),
                                                                                  ('Forest',
                                                                                   RandomForestRegressor(criterion='absolute_error',
                                                                                                         max_features='sqrt',
                                                                                                         min_samples_leaf=4))],
                                                                      n_jobs=-1,
                                                                      weights=[1,
                                                                               2]))]),
                           transformer=PowerTransformer())</pre></div></div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="dc27b7ca-c22e-4e1e-8f12-dee5b24180b4" type="checkbox" ><label class="sk-toggleable__label" for="dc27b7ca-c22e-4e1e-8f12-dee5b24180b4">rfe_preprocessor: Pipeline</label><div class="sk-toggleable__content"><pre>Pipeline(steps=[('preprocessor',
                 ColumnTransformer(n_jobs=-1,
                                   transformers=[('onehot_org',
                                                  Pipeline(steps=[('SI',
                                                                   SimpleImputer(fill_value='missing',
                                                                                 strategy='constant')),
                                                                  ('OHE_Org',
                                                                   OneHotEncoder(categories=[['College/University',
                                                                                              'High '
                                                                                              'School',
                                                                                              'Other '
                                                                                              'Team/Club']],
                                                                                 drop='first',
                                                                                 handle_unknown='ignore',
                                                                                 sparse=False))]),
                                                  ['typeOrganizationFrom']),
                                                 ('onehot...
                                                  Pipeline(steps=[('ratio_pipeline',
                                                                   Pipeline(steps=[('KNNI',
                                                                                    KNNImputer(n_neighbors=1)),
                                                                                   ('SS',
                                                                                    StandardScaler()),
                                                                                   ('PT',
                                                                                    PowerTransformer())])),
                                                                  ('pca',
                                                                   PCA(n_components=3,
                                                                       random_state=8))]),
                                                  ['BLK', 'FG', 'STL', 'AST',
                                                   '2P%', 'WS', 'G', 'WS/48',
                                                   'GS', 'FG%', 'ORB', 'FT',
                                                   'PTS', 'MP', 'PF', 'TOV',
                                                   '2P', 'DRB', 'eFG%'])])),
                ('rfe',
                 RFE(estimator=RandomForestRegressor(),
                     n_features_to_select=12))])</pre></div></div></div><div class="sk-serial"><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="ae846947-2077-4eda-91a2-feabd2f65ab8" type="checkbox" ><label class="sk-toggleable__label" for="ae846947-2077-4eda-91a2-feabd2f65ab8">preprocessor: ColumnTransformer</label><div class="sk-toggleable__content"><pre>ColumnTransformer(n_jobs=-1,
                  transformers=[('onehot_org',
                                 Pipeline(steps=[('SI',
                                                  SimpleImputer(fill_value='missing',
                                                                strategy='constant')),
                                                 ('OHE_Org',
                                                  OneHotEncoder(categories=[['College/University',
                                                                             'High '
                                                                             'School',
                                                                             'Other '
                                                                             'Team/Club']],
                                                                drop='first',
                                                                handle_unknown='ignore',
                                                                sparse=False))]),
                                 ['typeOrganizationFrom']),
                                ('onehot_pos',
                                 Pipeline(steps=[('SI',
                                                  Simp...
                                                 ('PT', PowerTransformer())]),
                                 ['Age', '3P%', 'PER', '3P', 'FT%']),
                                ('pca',
                                 Pipeline(steps=[('ratio_pipeline',
                                                  Pipeline(steps=[('KNNI',
                                                                   KNNImputer(n_neighbors=1)),
                                                                  ('SS',
                                                                   StandardScaler()),
                                                                  ('PT',
                                                                   PowerTransformer())])),
                                                 ('pca',
                                                  PCA(n_components=3,
                                                      random_state=8))]),
                                 ['BLK', 'FG', 'STL', 'AST', '2P%', 'WS', 'G',
                                  'WS/48', 'GS', 'FG%', 'ORB', 'FT', 'PTS',
                                  'MP', 'PF', 'TOV', '2P', 'DRB', 'eFG%'])])</pre></div></div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="12dc0ddb-d247-460f-aa11-a331095fc17c" type="checkbox" ><label class="sk-toggleable__label" for="12dc0ddb-d247-460f-aa11-a331095fc17c">onehot_org</label><div class="sk-toggleable__content"><pre>['typeOrganizationFrom']</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="bbe04287-a2a5-455f-8736-9d2ca29bc0fb" type="checkbox" ><label class="sk-toggleable__label" for="bbe04287-a2a5-455f-8736-9d2ca29bc0fb">SimpleImputer</label><div class="sk-toggleable__content"><pre>SimpleImputer(fill_value='missing', strategy='constant')</pre></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="702f3c6f-be85-48d9-9a33-1bd59346135c" type="checkbox" ><label class="sk-toggleable__label" for="702f3c6f-be85-48d9-9a33-1bd59346135c">OneHotEncoder</label><div class="sk-toggleable__content"><pre>OneHotEncoder(categories=[['College/University', 'High School',
                           'Other Team/Club']],
              drop='first', handle_unknown='ignore', sparse=False)</pre></div></div></div></div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="cf427468-7237-4b07-bee5-49e3eed7826a" type="checkbox" ><label class="sk-toggleable__label" for="cf427468-7237-4b07-bee5-49e3eed7826a">onehot_pos</label><div class="sk-toggleable__content"><pre>['Pos']</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="fefa0278-f86d-4c19-8fab-5149b6b15652" type="checkbox" ><label class="sk-toggleable__label" for="fefa0278-f86d-4c19-8fab-5149b6b15652">SimpleImputer</label><div class="sk-toggleable__content"><pre>SimpleImputer(fill_value='missing', strategy='constant')</pre></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="42ab8860-a7c6-44b4-9eea-a850ca1aa4f4" type="checkbox" ><label class="sk-toggleable__label" for="42ab8860-a7c6-44b4-9eea-a850ca1aa4f4">OneHotEncoder</label><div class="sk-toggleable__content"><pre>OneHotEncoder(categories=[['PG', 'SG', 'PF', 'SF', 'C']], drop='if_binary',
              handle_unknown='ignore', sparse=False)</pre></div></div></div></div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="6dbbae0c-66b4-4a68-8fd2-603e5e17ae44" type="checkbox" ><label class="sk-toggleable__label" for="6dbbae0c-66b4-4a68-8fd2-603e5e17ae44">target</label><div class="sk-toggleable__content"><pre>['nameOrganizationFrom', 'Tm']</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="affde898-c868-4a2e-b610-7e76db7c2db6" type="checkbox" ><label class="sk-toggleable__label" for="affde898-c868-4a2e-b610-7e76db7c2db6">TargetEncoder</label><div class="sk-toggleable__content"><pre>TargetEncoder(drop_invariant=True, handle_missing='return_nan', smoothing=5.0)</pre></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="baf897df-e0ea-4b20-a692-7b222cd5be9b" type="checkbox" ><label class="sk-toggleable__label" for="baf897df-e0ea-4b20-a692-7b222cd5be9b">KNNImputer</label><div class="sk-toggleable__content"><pre>KNNImputer(n_neighbors=2)</pre></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="752ba0fc-7ed4-41d6-822e-f0168acdfc48" type="checkbox" ><label class="sk-toggleable__label" for="752ba0fc-7ed4-41d6-822e-f0168acdfc48">StandardScaler</label><div class="sk-toggleable__content"><pre>StandardScaler()</pre></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="b2a17341-cc97-4d6b-af9b-26ad4c304380" type="checkbox" ><label class="sk-toggleable__label" for="b2a17341-cc97-4d6b-af9b-26ad4c304380">PowerTransformer</label><div class="sk-toggleable__content"><pre>PowerTransformer()</pre></div></div></div></div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="18cb696d-8c07-41d3-a570-b51b1a095a69" type="checkbox" ><label class="sk-toggleable__label" for="18cb696d-8c07-41d3-a570-b51b1a095a69">ordinal</label><div class="sk-toggleable__content"><pre>['numberPickOverall']</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="38b91823-e36e-496f-b11b-0954c649978d" type="checkbox" ><label class="sk-toggleable__label" for="38b91823-e36e-496f-b11b-0954c649978d">OrdinalEncoder</label><div class="sk-toggleable__content"><pre>OrdinalEncoder(categories=[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                            11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0,
                            19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0,
                            27.0, 28.0, 29.0, 30.0, ...]],
               handle_unknown='use_encoded_value', unknown_value=nan)</pre></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="bd8f208d-7daa-41f1-a3df-2567868ea9f5" type="checkbox" ><label class="sk-toggleable__label" for="bd8f208d-7daa-41f1-a3df-2567868ea9f5">KNNImputer</label><div class="sk-toggleable__content"><pre>KNNImputer(n_neighbors=1)</pre></div></div></div></div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="e49c9fad-9f43-40f5-ad10-36f6e58e3440" type="checkbox" ><label class="sk-toggleable__label" for="e49c9fad-9f43-40f5-ad10-36f6e58e3440">ratio</label><div class="sk-toggleable__content"><pre>['Age', '3P%', 'PER', '3P', 'FT%']</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="bda2633e-b291-436a-8035-4531b43a66ce" type="checkbox" ><label class="sk-toggleable__label" for="bda2633e-b291-436a-8035-4531b43a66ce">KNNImputer</label><div class="sk-toggleable__content"><pre>KNNImputer(n_neighbors=1)</pre></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="b54fd79c-1231-41f1-8859-5dbd7e773d4b" type="checkbox" ><label class="sk-toggleable__label" for="b54fd79c-1231-41f1-8859-5dbd7e773d4b">StandardScaler</label><div class="sk-toggleable__content"><pre>StandardScaler()</pre></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="9c49630c-a33c-4cf6-803b-ca6b8188917d" type="checkbox" ><label class="sk-toggleable__label" for="9c49630c-a33c-4cf6-803b-ca6b8188917d">PowerTransformer</label><div class="sk-toggleable__content"><pre>PowerTransformer()</pre></div></div></div></div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="b0afc4ab-8779-4bd1-8147-6f684485ee88" type="checkbox" ><label class="sk-toggleable__label" for="b0afc4ab-8779-4bd1-8147-6f684485ee88">pca</label><div class="sk-toggleable__content"><pre>['BLK', 'FG', 'STL', 'AST', '2P%', 'WS', 'G', 'WS/48', 'GS', 'FG%', 'ORB', 'FT', 'PTS', 'MP', 'PF', 'TOV', '2P', 'DRB', 'eFG%']</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="6c9bc248-d4de-4987-9add-1eaac85dd645" type="checkbox" ><label class="sk-toggleable__label" for="6c9bc248-d4de-4987-9add-1eaac85dd645">ratio_pipeline: Pipeline</label><div class="sk-toggleable__content"><pre>Pipeline(steps=[('KNNI', KNNImputer(n_neighbors=1)), ('SS', StandardScaler()),
                ('PT', PowerTransformer())])</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="7ddb92f8-53a0-4efd-8522-362609ac665f" type="checkbox" ><label class="sk-toggleable__label" for="7ddb92f8-53a0-4efd-8522-362609ac665f">KNNImputer</label><div class="sk-toggleable__content"><pre>KNNImputer(n_neighbors=1)</pre></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="396ebe9e-e4a0-41ff-ab6f-1273d7912090" type="checkbox" ><label class="sk-toggleable__label" for="396ebe9e-e4a0-41ff-ab6f-1273d7912090">StandardScaler</label><div class="sk-toggleable__content"><pre>StandardScaler()</pre></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="76d0058a-9942-4a4a-9f23-da9031eb61a7" type="checkbox" ><label class="sk-toggleable__label" for="76d0058a-9942-4a4a-9f23-da9031eb61a7">PowerTransformer</label><div class="sk-toggleable__content"><pre>PowerTransformer()</pre></div></div></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="f6035e23-2034-4edf-9095-96dac36f6ce6" type="checkbox" ><label class="sk-toggleable__label" for="f6035e23-2034-4edf-9095-96dac36f6ce6">PCA</label><div class="sk-toggleable__content"><pre>PCA(n_components=3, random_state=8)</pre></div></div></div></div></div></div></div></div></div></div><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="2ec6e95d-6d29-4dad-b76f-ebfea1e1299f" type="checkbox" ><label class="sk-toggleable__label" for="2ec6e95d-6d29-4dad-b76f-ebfea1e1299f">rfe: RFE</label><div class="sk-toggleable__content"><pre>RFE(estimator=RandomForestRegressor(), n_features_to_select=12)</pre></div></div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="69913cde-0f71-4d45-88f6-e972a35e8b46" type="checkbox" ><label class="sk-toggleable__label" for="69913cde-0f71-4d45-88f6-e972a35e8b46">RandomForestRegressor</label><div class="sk-toggleable__content"><pre>RandomForestRegressor()</pre></div></div></div></div></div></div></div></div></div></div><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="881e1920-c7ec-44d6-8f19-72200e1a20ed" type="checkbox" ><label class="sk-toggleable__label" for="881e1920-c7ec-44d6-8f19-72200e1a20ed">regressor: VotingRegressor</label><div class="sk-toggleable__content"><pre>VotingRegressor(estimators=[('GB',
                             GradientBoostingRegressor(criterion='squared_error',
                                                       max_features='auto')),
                            ('Forest',
                             RandomForestRegressor(criterion='absolute_error',
                                                   max_features='sqrt',
                                                   min_samples_leaf=4))],
                n_jobs=-1, weights=[1, 2])</pre></div></div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><label>GB</label></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="dac10eea-6561-43ff-abed-c05960fa7586" type="checkbox" ><label class="sk-toggleable__label" for="dac10eea-6561-43ff-abed-c05960fa7586">GradientBoostingRegressor</label><div class="sk-toggleable__content"><pre>GradientBoostingRegressor(criterion='squared_error', max_features='auto')</pre></div></div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><label>Forest</label></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="acf62520-c75a-4f0e-96d3-ae049249df6c" type="checkbox" ><label class="sk-toggleable__label" for="acf62520-c75a-4f0e-96d3-ae049249df6c">RandomForestRegressor</label><div class="sk-toggleable__content"><pre>RandomForestRegressor(criterion='absolute_error', max_features='sqrt',
                      min_samples_leaf=4)</pre></div></div></div></div></div></div></div></div></div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="765e1e8b-02cd-4d66-bd4e-a42bcded4b47" type="checkbox" ><label class="sk-toggleable__label" for="765e1e8b-02cd-4d66-bd4e-a42bcded4b47">PowerTransformer</label><div class="sk-toggleable__content"><pre>PowerTransformer()</pre></div></div></div></div></div></div></div></div></div></div>


## Skills
Technical skills honed in this project are:
- Batch-Wrangling of csv-files
- Feature Engineering
- Pandas Data Type Optimization
- Understanding of Pandas & NumPy internals
- Using different flat file types with faster I/O

## Personal Learnings:
This time the personal learnings can be summarized in a checklist for future projects:
1. **Use the pd.read_csv() parameters**:
  - **usecols**: read only the specified columns
  - **nrows**: read only the rows needed
  - **dtype**: if you already know the optimal data type for the columns, specify it with a dict
  - **chunksize**: load the data as an iterator and preprocess/aggregate it in a loop, concatenating the results again

2. **Choose correct data types**: <br/>
**df.convert_dtypes()** is a really powerful method that converts the df into a df with correct data types. For example: There is a column that has a float64 data type, but it only holds numbers with .0, so it's basically all integers. This method corrects the column automatically to int64. Or Object data types that are actually strings. All this safes memory and it also brings another advantage: the new data types are pandas ExtensionDtype. This data type allows integer columns to have NA-values and not convert to float, like it used to be. With the [Pandas Version Update 1.3.0](https://pandas.pydata.org/docs/whatsnew/v1.3.0.html) from July 2021 those data types can be downcasted.

3. **Downcast numeric values**: <br/>
When pandas reads a csv it looks at the first few rows for each column and then guesses the data type (int, float, string, etc.). Since pandas doesn't know if the last value in this column exceeds any size (big numbers), it automatically upcasts to the biggest format (int64, float64, etc.) which is often not needed. You can use the **pd.to_numeric(downcast=str)** function to take care of this.

4. **Use the categorical data type for low cardinality strings**: <br/>
Often Strings are represented as objects in a pandas DataFrame, which itself can be a memory safer when transformed to strings. The problem with strings though is that they occupy a lot of space and are often frequently repeated in a column. This means the column has low cardinality and a more efficient data storage option are categories. Much like OrdinalEncoder in Scikit-learn, pandas safes a mapping for all the different strings internally and the rows get the corresponding number (internally). That way strings won't be repeated and memory is safed --> **df.astype('category')**

5. **Use sparse arrays for low cardinality numbers**: <br/>
Sparse data is data which contains mostly NaN / missing value, though any value can be chosen, including 0. On the contrary, A column in which the majority of elements are non zero is called dense. Convert to sparse with **df[column] = df[column].astype(pd.SparseDtype(df['FG'].dtype, pd.NA))** or directly use the parameter of pandas dummy function: **pd.get_dummies(data, sparse=True)**. Though this technique was not needed in this project, it could be an advantage in future projects with sparse data.

6. **Using optimized I/O file formats**: <br/>
A csv-file doesn't save any data types so it would be a pitty to loose all the work of optimizing a dataframe when saved to csv. There are more suitable file formats with faster I/O as well, like: pickle, hdf5, feather, parquet, etc. Just use pandas **df.to_feather(filepath, compression='lz4')**.

## Outlook
There is still one big disadvantage: The function is only applicable when the DataFrame is already loaded into RAM. So what if we need to optimize the data types before loading it into RAM because it doesn't fit in there yet? That's where chunking and saving the minimum and maximum of the column into a DataFrame comes in handy. One could then downcast this intermediate DataFrame and safe the optimized Data Types as a dictionary for the read_csv parameter "dtype: dict". Another opportunity could be a out-of-core library like [Vaex](https://vaex.io/docs/index.html) that makes use of memory-mapping and doesn't load all the data into RAM.

Also sparse arrays weren't used in this project and could be an advantage in certain datasets with sparse data, e.g. tables for recommendation engines that have a lot of 1/0 data. This feature would exceed an acceptable function length and could be added in a future project for a custom module.


## References
**Downcasting and Categoricals**:
- [Dataquest Blog](https://www.dataquest.io/blog/pandas-big-data/)
- [Pandas Documentation](https://pandas.pydata.org/pandas-docs/stable/user_guide/scale.html#)

**Sparse Data**:
- [TDS Blog](https://towardsdatascience.com/working-with-sparse-data-sets-in-pandas-and-sklearn-d26c1cfbe067)

**Optimal I/O Files**:
- [TDS Blog](https://towardsdatascience.com/the-best-format-to-save-pandas-data-414dca023e0d)

The **Dataset** is from Ken Huang: 
- [Kaggle Profile](https://www.kaggle.com/kenhuang41/nba-basic-game-data-by-player)


## 📫 Let's connect and learn from each other:

[<img src="https://github.com/kevin-goetz/kevin-goetz/blob/main/LinkedIn Logo.png" height="40em" align="center" alt="Connect with Me on LinkedIn" title="Connect with Me on LinkedIn"/>](https://linkedin.com/in/kgötz) [<img src="https://github.com/kevin-goetz/kevin-goetz/blob/main/Codewars Logo.svg" height="40em" align="center" alt="Connect with Me on Codewars" title="Connect with Me on Codewars"/>](https://www.codewars.com/users/kevin-goetz)


