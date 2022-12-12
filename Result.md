## Regression Models
- Scaling methods: None, StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, Normalizer
- Score method: r2, f1 (Logistic Regression)
- Score is calculated by:
  - Choose Scaling method then predict 100 times
  - Get the Mean value of 100 times prediction
  > Score is the highest Mean of 100 times prediction of each Scaling methods

### Logistic Regression
- Best Scaler:  RobustScaler
- Score:  0.9

### Support Vector Regression
- Best Scaler:  MaxAbsScaler
- Score:  0.784

### SGD Regression
- Best Scaler:  MinMaxScaler
- Score:  0.776

### Random Forest
- Best Scaler:  None
- Score:  0.769

### Ridge Regression
- Best Scaler:  MinMaxScaler
- Score:  0.759

### Gaussian Regression
- Best Scaler:  MinMaxScaler
- Score:  0.759

### K-Nearest Neighbors Regression 
- Best Scaler:  None
- Score:  0.757

###  Linear Regression
- Best Scaler:  MinMaxScaler
- Score:  0.743

### Huber Regression
- Best Scaler:  MaxAbsScaler
- Score:  0.729

### Lasso Regression
- Best Scaler:  None
- Score:  0.63

### Decision Tree Regression
- Best Scaler:  MaxAbsScaler
- Score:  0.517
