---
layout: post
title: "Pandas and sklearn's pipelines"
date: 2018-02-12
---

Having to deal with a lot of labeled data, one won't come around using the great [pandas](https://pandas.pydata.org/) library sooner or later.
The benefits of it over raw numpy are obvious.
Now pandas is a library that came up some time after numpy.
Bad thing about this - some other great tools started growing immensely without having pandas at hand so they had to be built upon numpy.

With that I am talking about [sklearn](http://scikit-learn.org/stable/index.html) and in particular their awesome mechanisms for [pipelines](http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) and [feature unions](http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.FeatureUnion.html#sklearn.pipeline.FeatureUnion).
While most of the people involved in Data Science know what sklearn is, few have actually used the latter.
One of the reasons leading to this collective unused potential might be the lack of a pandas support.

Creating a pipeline is as simple as this:

{% highlight py %}
import pandas as pd
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline, make_pipeline

# create pipeline
pipe = Pipeline([('scaler', StandardScaler()),
                 ('polynomials', PolynomialFeatures()),
                ])

# alternatively
pipe = make_pipeline(StandardScaler(), PolynomialFeatures())

{% endhighlight %}

Transforming data through this pipeline is also straight forward:

{% highlight py %}
df = pd.read_csv('path/to/data.csv')

transformed = pipe.fit_transform(df)

{% endhighlight %}

The 'fit_transform' function is just a consecutive call of 'fit()' and 'transform()', the same syntax as it is for sklearn's famous regression and classifier models.
You get the idea of pipelines.

Unfortunately there is a considerable drawback.

# Problem

