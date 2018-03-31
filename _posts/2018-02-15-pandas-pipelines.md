---
layout: post
title: "Pandas and sklearn pipelines"
date: 2018-02-15
---
Having to deal with a lot of labeled data, one won't come around using the great [pandas](https://pandas.pydata.org/) library sooner or later.
The benefits of it over raw numpy are obvious.

Now pandas is a library that came up some time after numpy.
Bad thing about this - some other great tools started growing immensely without having pandas at hand so they had to be built upon numpy.

With that I am talking about [sklearn](http://scikit-learn.org/stable/index.html) and in particular their awesome mechanisms for [pipelines](http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) and [feature unions](http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.FeatureUnion.html#sklearn.pipeline.FeatureUnion).
While most of the people involved in Data Science know what sklearn is, few have actually used the latter.
One of the reasons leading to this collective unused potential might be the lack of a pandas support.

> Here's a brief refresher on pipelines. Creating a pipeline is as simple as this:

 ```py
import pandas as pd
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline, make_pipeline

# create pipeline
pipe = Pipeline([('scaler', StandardScaler()),
                 ('polynomials', PolynomialFeatures()),
                ])

# alternatively
pipe = make_pipeline(StandardScaler(), PolynomialFeatures())

```

> Transforming data through this pipeline is also straight forward:

```py
df = pd.read_csv('path/to/data.csv')
transformed = pipe.fit_transform(df)
```

> The **fit_transform** function is just a consecutive call of **fit()** and **transform()**, the same syntax as it is for sklearn's famous regression and classifier models.
> You get the idea of pipelines.
>
> Unfortunately there is a considerable drawback.

# Problem

The returned object of pipelines and especially feature unions are numpy arrays. This is partly due to the internals of pipelines and partly due to the elements of the pipeline themselves, that is, sklearn's statistical models and transformers such as *StandardScaler*.
When you rely on your transformed dataset to retain the pandas dataframe structure e.g. because your model has a column-specific processing, there is no out-of-the-box solution.

Please note, that there is a wrapper available, called [sklearn-pandas](https://github.com/scikit-learn-contrib/sklearn-pandas).
Before I came up with the following solution I tried that wrapper and I felt it has some implausible limitations such as no pandas output in case of a default transformation or a very cluttered syntax among others.
I kind of lost control over some pipelines when mixed with feature unions so that special handlings weren't feasible anymore.

# Solution

Facing some certain challenges in my own personal machine learning routines that I couldn't tackle with the beforementioned wrapper I found a homebrew approach to be not that complicated as it might seem.
Moreover, with the following blueprints I gained full control over each element in a pipeline that can be arbitrarily mixed with feature unions and I still have a pandas dataframe as returned object.

Let's dive right into it!

## Transformation Chains

Each pipeline and feature union consists of elements, their chains, that work on top of each other or in parallel, hand in hand.
Here's a compiled list of some transformers that are pandas-aware.

### Simple Transformations

Imagine you want your dataset to be expanded with the data itself but transformed e.g. by taking the square root of it or by the log of it.
You would approach this by a simple feature union but what are the elements of that? Examine the following:

```py
class SimpleTransformer(BaseEstimator, TransformerMixin):
    """Apply given transformation."""
    def __init__(self, trans_func, untrans_func, columns):
        self.transform_func = trans_func
        self.inverse_transform_func = untrans_func
        self.cols = columns

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        x = self._get_selection(x)
        return self.transform_func(x) if callable(self.transform_func) else x

    def inverse_transform(self, x):
        return self.inverse_transform_func(x) \
            if callable(self.inverse_transform_func) else x

    def _get_selection(self, df):
        assert isinstance(df, pd.DataFrame)
        return df[self.cols]

    def get_feature_names(self):
        return self.cols
```
Each class you intent to put in a pipeline or feature union should inherit from **BaseEstimator** and **TransformerMixin**.
While the former makes your class accessible for hyper parameter methods such as [GridSearch](http://scikit-learn.org/stable/modules/grid_search.html) the latter applies a robust **fit_transform** function.
For the base classes the only mandatory functions are **fit** and **transform** but with **__init__** you can give your transformer some config at initialization and the **inverse__transform** enables the inverse transform call on the full pipeline if each element supports it. Furthermore, the **get_feature_names** function is crucial when you want the names of transformed features in a feature union accessible, which is a key feature as we'll see later.

Let's take a closer look at the content.
This class gets a transformation and untransformation instruction for transform() and inverse_transform() respectively.
Besides that, at init we can specify the certain columns that need to be transformed.
Note, that this can be arbitrarily adapted to your own needs.
If you need the processing of all columns given, just omit the first line in the transform function.

So far so good.
When expanding my dataset I often want the actual dataset also be present in my feature union, which is not retained by default.
We can work around this by having an identity transformer.
We can even use our new class **SimpleTransformer** for that - just pass *None* for trans_func and untrans_func.

A very basic feature union that expects the pandas dataframe format can look like this:
```py
from sklearn.pipeline import FeatureUnion
import numpy as np

all_feature_names = ['Age', 'Gender', 'Height', 'Weight', 'y1', 'y2']

simple_union = FeatureUnion([('simple_trans_y',
                               SimpleTransformer(np.sqrt, np.square,
                                                 ['y1', 'y2'])
                              ),
                              ('identity',
                               SimpleTransformer(None, None,
                                ['Age', 'Gender', 'Height', 'Weight'])
                              )
                             ])
```
Don't be confused by all those brackets - the **FeatureUnion** class just takes as argument a list of tuples where each tuple consists of a name and a transformer.
Here, a dataset is simply filtered by subsets of **all_feature_names** where the columns *y1* and *y2* are transformed into their square root (assume their content to be floating point numbers).
Well, the output of this union will still be a numpy matrix, but hold on, the magic is still to come.
So far we have seen how to have transformations specific to certain columns but the df format is not retained yet.
Let's examine a few more examples before we get to this next step.

### Scaling Transformations

The classic. Scaling your dataset by a **StandardScaler** or **MinMaxScaler** is what data scientists do for a living since a lot of linear models rely on this preprocessing in order to *learn* the latent patterns.
But what if we want only specific columns to be scaled?
What if I need a separate scaling for my independent and dependent features since I want to inverse the scaling of my predictions later that naturally embrace only the target variables?

```py
class Scaler(BaseEstimator, TransformerMixin):
    """scales selected columns only with given scaler"""
    def __init__(self, scaler, columns):
        self.scaler = scaler
        self.cols = columns

    def fit(self, X, y=None):
        X = self._get_selection(X)
        self.scaler.fit(X, y)
        return self

    def transform(self, X):
        X = self._get_selection(X)
        return self.scaler.transform(X)

    def inverse_transform(self, X):
        return self.scaler.inverse_transform(X)

    def _get_selection(self, df):
        assert isinstance(df, pd.DataFrame)
        return df[self.cols]

    def get_feature_names(self):
        return self.cols
```
This scaler class is pretty straight forward and calls in each of its functions the corresponding scaler function, where the scaler was given during initialization.
The difference to the SimpleTransformer is that we now also do something during fitting but no surprises here.
Build the feature union like this:

```py
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion

scaling_union = FeatureUnion([('scaler_x',
                              Scaler(StandardScaler(),
                                     ['Age', 'Gender', 'Height', 'Weight']),
                              ('scaler_y',
                               Scaler(StandardScaler(),
                                      ['y1', 'y2']))
                             ])
```
You get the idea.

### Rolling Transformations

You work with time series data and want your dataset expanded with rolling statistics?
No problem, just consider this:

```py
class RollingFeatures(BaseEstimator, TransformerMixin):
    """This Transformer adds rolling statistics"""
    def __init__(self, columns, lookback=10):
        self.lookback = lookback
        self.cols = columns
        self.transformed_cols = None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = self._get_selection(X)
        feat_d = {'std': X.rolling(self.lookback).std(),
                  'mean': X.rolling(self.lookback).mean(),
                  'sum': X.rolling(self.lookback).sum()
                  }
        for k in feat_d:
            feat_d[k].columns = \
                ['{}_rolling{}_{}'.format(c, self.lookback, k) for
                 c in X.columns]
        df = pd.concat(list(feat_d.values()), axis=1)
        self.transformed_cols = list(df.columns)
        return df

    def _get_selection(self, df):
        assert isinstance(df, pd.DataFrame)
        return df[self.cols]

    def get_feature_names(self):
        return self.transformed_cols
```

Here we come up with completely new feature names the first time.
Note, that we have a new class variable **transformed_cols** to take account of those cols that were generated here.

### Cleaning the DataFrame

As simple as this:

```py
class DFCleaner(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        X.dropna(inplace=True)
        X.reset_index(drop=True, inplace=True)
        return X
```

By now you should have realized, that the possibilities are endless.
Every transformation you would apply on your data is ultimately convertible into a transformer class.
Now let's see how to put them into shape such that these chains are actually stackable.

## Retain the DataFrame format for the output

For this challenge we can exploit the following simple trick.
The **FeatureUnion** class has a method called **get_feature_names** that exhibits the feature names of each transformer *although their output is a numpy matrix*.
In order to workaround the numpy output we can make each feature union a two-step pipeline where the union denotes the first step while a transformer fetching the actual feature names represents the second step.
Sounds crazy?
Check this out:

```py
class FeatureUnionReframer(BaseEstimator, TransformerMixin):
    """Transforms preceding FeatureUnion's output back into Dataframe"""
    def __init__(self, feat_union, cutoff_transformer_name=True):
        self.union = feat_union
        self.cutoff_transformer_name = cutoff_transformer_name

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, np.ndarray)
        if self.cutoff_transformer_name:
            cols = [c.split('__')[1] for c in self.union.get_feature_names()]
        else:
            cols = self.union.get_feature_names()
        df = pd.DataFrame(data=X, columns=cols)
        return df

    @classmethod
    def make_df_retaining(cls, feature_union):
        """With this method a feature union will be returned as a pipeline
        where the first step is the union and the second is a transformer that
        re-applies the columns to the union's output"""
        return Pipeline([('union', feature_union),
                         ('reframe', cls(feature_union))])
```
This class does the job.
The optional bool argument gives you the freedom to keep the default prefix in feature union chains that is the name of the transformer.
The class' static method is making things even more comfortable as we don't need to instantiate it explicitly.

See the following example where all falls into place:

```py
from sklearn.pipeline import make_union, make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np

x_feats = ['Age', 'Gender', 'Height', 'Weight']
y_feats = ['y1', 'y2']

featurize_union = make_union(SimpleTransformer(np.sqrt, np.square, y_feats),
                             SimpleTransformer(None, None, x_feats),
                             RollingFeatures(x_feats, lookback=10)
                             )
scaling_union = make_union(Scaler(StandardScaler(), x_feats),
                           Scaler(StandardScaler(), y_feats)
                          )

featurize_pipe = FeatureUnionReframer.make_df_retaining(featurize_union)
scaling_pipe = FeatureUnionReframer.make_df_retaining(scaling_union)

pipe = make_pipeline(featurize_pipe,
                     DFCleaner(),
                     scaling_pipe)

```

The output of **pipe** will be our dataset as Pandas DataFrame with all transformations applied. Awesome!

# Conclusion

The presented concepts are arbitrarily expansible and give full control over the transformations that are applied on the dataset while the output is retained as DataFrame.
All key features remain such as hyper parameter optimization with **GridSearch** or avoidance of testset leakage through fit() and transform().

# Bonus

You want to train Keras LSTMs on time series data and you need the dataset reordered for batch training?
At the same time, the set must be back-convertible for a comparison of the prediction to the actual target data?
Fear not, this will help:

```py
class ReSamplerForBatchTraining(BaseEstimator, TransformerMixin):
    """This transformer sorts the samples according to a
    batch size for batch training"""
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.indices, self.columns = [], []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        # cut the tail
        trunc_idx = len(X) % self.batch_size
        X = X.iloc[:-trunc_idx, :]

        # reorder
        new_idcs = np.tile(np.arange(self.batch_size), len(X) //
                           self.batch_size)
        assert len(X) == new_idcs.shape[0], \
            "{} != {}".format(len(X), new_idcs.shape[0])
        X['new_idx'] = new_idcs
        X.sort_values(by='new_idx', ascending=True, inplace=True)
        self.indices = X.index
        X.reset_index(drop=True, inplace=True)
        X.drop(['new_idx'], axis=1, inplace=True)
        self.columns = X.columns
        return X

    def inverse_transform(self, X):
        # columns undefined
        inversed = pd.DataFrame(X, index=self.indices).sort_index()
        return inversed
```

You're welcome.

* * *

