EEG-Based Explainable AI Framework for Tic Disorder Classification and Severity Prediction
1.Overview
ETD-XAI_code is a modular and reproducible Python framework for EEG-based machine learning and explainable artificial intelligence (XAI), designed for tic disorder (TD) classification and severity prediction.
The framework integrates multidomain EEG feature extraction, statistical analysis, machine learning model evaluation, feature selection, and SHAP-based interpretability, enabling clinically meaningful and transparent neurophysiological analysis.
2.Key Features
Multidomain EEG feature extraction (time, frequency, time–frequency)
Channel-wise feature representation (19 EEG channels)
Statistical group comparison (TD vs HC)
RFECV-based feature selection with AUROC validation
Cross-validated model evaluation
ROC curve analysis for classification performance
SHAP-based explainability:
Global feature importance
Top feature ranking
Severity grading (beeswarm plots)
Cross-modal interaction interpretation
Fully modular and reproducible scripts
Repository Structure
Age-Analysis.py
Demographic and subgroup statistical analysis.
cross-validated AUROC during RFECV-based feature selection.py
Feature selection with RFECV and AUROC evaluation.
Receiver operating characteristic (ROC) curves for first-stage tic disorder classification models.py
ROC analysis for TD vs HC classification models.
Global SHAP feature importance.py
Global feature importance based on SHAP values.
Global SHAP feature importance (Top 20).py
Visualization of top-ranked features.
SHAP beeswarm plot for the second-stage severity grading model.py
SHAP interpretation for severity prediction model.
SHAP value distributions of cross-modal interaction features.py
Cross-modal feature interaction analysis using SHAP.
Data Description
EEG multidomain features (time / frequency / time–frequency)
19-channel EEG recordings
Subject-level feature matrix (channel-wise concatenation)
3.Workflow
EEG preprocessing and feature extraction
Construction of 19 × 47 channel-wise feature matrix
Statistical comparison (Mann–Whitney U test + FDR correction)
RFECV-based feature selection
Classification model training and cross-validation
ROC curve evaluation
SHAP-based explainability analysis
Clinical interpretation of neurophysiological biomarkers
Explainability (XAI)
This framework uses SHAP to provide interpretable insights into:
Feature importance across EEG modalities
Spatial and spectral distribution of discriminative biomarkers
Cross-modal feature interactions
Severity-related neurophysiological patterns
4.Reproducibility
All scripts are independently executable and designed to ensure:
Transparent statistical analysis
Reproducible machine learning pipelines
Consistent feature selection procedures
Interpretable AI outputs
License
5.Apache License
Version 2.0, January 2004
http://www.apache.org/licenses/
Copyright (c) 2026 Wang Jianping
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.

