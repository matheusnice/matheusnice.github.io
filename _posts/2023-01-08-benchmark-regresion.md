---
title: "Benchmarking Regression"
date: 2023-01-08T10:00:00-03:00
categories:
  - blog
tags:
  - Machine Learning
  - Regression
---

Benchmarking regression algorithms using `sklearn` and `python`

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

Check out the [Jekyll docs][jekyll-docs] for more info on how to get the most out of Jekyll. File all bugs/feature requests at [Jekyll’s GitHub repo][jekyll-gh]. If you have questions, you can ask them on [Jekyll Talk][jekyll-talk].

[jekyll-docs]: https://jekyllrb.com/docs/home
[jekyll-gh]:   https://github.com/jekyll/jekyll
[jekyll-talk]: https://talk.jekyllrb.com/
