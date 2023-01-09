---
title: "Benchmarking Regression Models"
date: 2023-01-08T00:00:00-03:00
categories:
  - blog
tags:
  - Machine Learning
  - Regression
---

In this post, I'm going to show how we can create a benchmark to evaluate regression models in terms of runtime and quality of the results.
For this benchmark, we will use the `House Prices - Advanced Regression Techniques` dataset available [here][house-data].

This dataset contains both numeric and categorical features, but we're not going to worry about EDA or feature engineering, since our goal here is to provide a framework that can be used to evaluate different models.

For data preprocessing, let's simply create:
- A numeric pipeline that will fill the missing values with `0` and apply a `StandardScaler` transformer
- A categorical pipeline that will fill the missing values with `'NA'` and apply a `OneHotEncoder` transformer

```python
X = train_set.drop('SalePrice', axis=1)
y = train_set['SalePrice'].copy()

cat_cols = list(X.select_dtypes('object').columns)
num_cols = list(X.select_dtypes(np.number).columns)

num_pipe = Pipeline([
    ('num_imputer', SimpleImputer(strategy='constant', fill_value=0)),
    ('scaler', StandardScaler())
])

cat_pipe = Pipeline([
    ('cat_imputer', SimpleImputer(strategy='constant', fill_value='NA')),
    ('onehotencode', OneHotEncoder())
])

preprocessor = ColumnTransformer([
    ('num_pipe', num_pipe, num_cols),
    ('cat_pipe', cat_pipe, cat_cols)
])

X_tr = preprocessor.fit_transform(X)
```

Now we will define our benchmark that will evaluate the model using cross validation and return the results as a list.
The scoring metrics that were chosen here are the RMSE and R2

```python
cv = ShuffleSplit(n_splits=5, test_size=0.2)

def bench_regressor(model, X, y, cv):
    cv_results = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=['r2', 'neg_root_mean_squared_error']
    )
    r2 = cv_results['test_r2'].mean()
    rmse = -cv_results['test_neg_root_mean_squared_error'].mean()
    fit_time = cv_results['fit_time'].mean()
    name = model.__class__.__name__
    
    return [name, fit_time, rmse, r2]
```

Now, let's run the benchmark and compare four approaches
- LinearRegression
- DecisionTreeRegressor
- GradientBoostingRegressor
- RandomForestRegressor

```python
estimators = [
    LinearRegression(),
    DecisionTreeRegressor(),
    GradientBoostingRegressor(),
    RandomForestRegressor(),
]

results = []

for estimator in estimators:
    model_results = bench_regressor(estimator, X_tr, y, cv=cv)
    results.append(model_results)
```

Finally, let's plot and visualize the results.

```python
def plot_benchmark_results():
    
    df = pd.DataFrame(results, columns=['name', 'fit_time', 'rmse', 'r2'])
    
    fig, ax = plt.subplots(1, 3, figsize=(13,3), sharey=True)

    ax[0].grid()
    bars = ax[0].barh(df.name, df.fit_time, color='r', alpha=0.6)
    ax[0].set_title('fit time (s)')
    for bar in bars:
        width = bar.get_width()
        ax[0].text(
            width/2,
            bar.get_y() + bar.get_height()/2,
            f'{round(width, 1)}s',
            fontsize=11
        )

    ax[1].grid()
    bars = ax[1].barh(df.name, df.rmse, color='g', alpha=0.6)
    ax[1].set_title('RMSE')
    for bar in bars:
        width = bar.get_width()
        ax[1].text(
            width/2,
            bar.get_y() + bar.get_height()/2,
            f'{int(width)}',
            fontsize=11
        )

    ax[2].grid()
    bars = ax[2].barh(df.name, df.r2, alpha=0.6)
    ax[2].set_title('$R^2$')
    for bar in bars:
        width = bar.get_width()
        ax[2].text(
            width/2,
            bar.get_y() + bar.get_height()/2,
            f'{round(width, 1)}',
            fontsize=11
        )
```

Results:
<figure style="width: 1200px">
  <img src="{{ site.url }}{{ site.baseurl }}/assets/images/regression-plot.png" alt="">
  <figcaption>A plot with the benchmark results</figcaption>
</figure> 

For this particular test, we could see that the `GradientBoostingRegressor` was the model with best performance on both of the metrics evaluated.

The notebook with the full benchmark can be found [here][notebook].

[house-data]: https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques
[notebook]: https://github.com/matheusnice/benchmarking-regression-models/blob/main/benchmark-regression-models.ipynb