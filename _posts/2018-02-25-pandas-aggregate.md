---
layout: post
title: "High Cardinality and Custom Aggregations"
date: 2018-02-25
comments: true
---

Every now and then transforming data column-by-column, e.g. by adding the log of each floating-point feature to the dataset, is maybe not enough and more ambitious transformations are desired.
Perhaps you would like a certain feature transformed dependending on the value the corresponding observation holds in another feature.

# Pandas aggregate function
The Pandas library comes with a highly optimized aggregation function, that can be utilized with the most general statistical summaries by default.
```py
>>> import numpy as np
>>> import pandas as pd
>>> df = pd.DataFrame(np.random.randn(10, 3), columns=['A', 'B', 'C'],
                       index=pd.date_range('1/1/2000', periods=10))
>>> df['D'] = np.tile([0, 1], 5)
>>> df
                   A         B         C  D
2000-01-01  0.506738  0.063401 -0.864185  0
2000-01-02  0.067840  0.272242 -1.425997  1
2000-01-03  0.371210  0.558194 -0.151807  0
2000-01-04 -0.363128 -0.882355  1.254074  1
2000-01-05  0.735415 -0.072763 -0.634464  0
2000-01-06  0.796265  1.525935  0.086280  1
2000-01-07 -0.048933  0.178912 -0.039579  0
2000-01-08 -0.730741 -1.026607  1.095198  1
2000-01-09 -1.649780  1.245154 -0.610097  0
2000-01-10  0.003570 -0.132191 -0.196222  1

>>> df.agg(['min', 'max', 'sum'])
            A         B         C  D
min -1.649780 -1.026607 -1.425997  0
max  0.796265  1.525935  1.254074  1
sum -0.311544  1.729923 -1.486800  5

```
The *agg* function is short for aggregation and takes either strings of known function names such as *min* or *sum* or homebrewed customized aggregation functions.
One could also get these statistical characteristics by other means but the pandas aggregation is nevertheless worth a try since it runs with a c implementation in the background making it super fast.

# Categorical features with high cardinality
Imagine you are given a categorical feature with a very high cardinality, say, the postal code of a citizen encoded into 200 integer numbers from 0 to 199.
Postal codes have no greater-less relationship but the enumeration suggests it.
While decisiontree-based models are capable of ignoring the inevitable ordering of categorical feature encodings, usual linear models desperately fall for that.
The result is over-fitting.

```py
>>> # Assume 10 postal codes for this demo
>>> df['PostalCode'] = np.asarray(list(range(10)))
>>> df
                   A         B         C  D  Postalcode
2000-01-01  0.506738  0.063401 -0.864185  0           0
2000-01-02  0.067840  0.272242 -1.425997  1           1
2000-01-03  0.371210  0.558194 -0.151807  0           2
2000-01-04 -0.363128 -0.882355  1.254074  1           3
2000-01-05  0.735415 -0.072763 -0.634464  0           4
2000-01-06  0.796265  1.525935  0.086280  1           5
2000-01-07 -0.048933  0.178912 -0.039579  0           6
2000-01-08 -0.730741 -1.026607  1.095198  1           7
2000-01-09 -1.649780  1.245154 -0.610097  0           8
2000-01-10  0.003570 -0.132191 -0.196222  1           9

```

## One Hot Encoding
A popular way around is so-called one-hot-encoding, that is, converting the single categorical feature with n distinct values into n boolean features.
For low cardinality features, such as gender (male or female), this is a pretty neat thing but for our postal code we will end up with sparse features that bloats RAM and may even hinder statistical models to learn latent patterns as the feature space explodes.

```py
>>> # One hot encode
>>> encoded_df = pd.get_dummies(df['Postalcode'], prefix='postalcode_')
>>> encoded_df
            postalcode__0  postalcode__1  postalcode__2  postalcode__3  \
2000-01-01              1              0              0              0   
2000-01-02              0              1              0              0   
2000-01-03              0              0              1              0   
2000-01-04              0              0              0              1   
2000-01-05              0              0              0              0   
2000-01-06              0              0              0              0   
2000-01-07              0              0              0              0   
2000-01-08              0              0              0              0   
2000-01-09              0              0              0              0   
2000-01-10              0              0              0              0   

            postalcode__4  postalcode__5  postalcode__6  postalcode__7  \
2000-01-01              0              0              0              0   
2000-01-02              0              0              0              0   
2000-01-03              0              0              0              0   
2000-01-04              0              0              0              0   
2000-01-05              1              0              0              0   
2000-01-06              0              1              0              0   
2000-01-07              0              0              1              0   
2000-01-08              0              0              0              1   
2000-01-09              0              0              0              0   
2000-01-10              0              0              0              0   

            postalcode__8  postalcode__9  
2000-01-01              0              0  
2000-01-02              0              0  
2000-01-03              0              0  
2000-01-04              0              0  
2000-01-05              0              0  
2000-01-06              0              0  
2000-01-07              0              0  
2000-01-08              0              0  
2000-01-09              1              0  
2000-01-10              0              1  
```

## Alternatives to One-Hot-Encoding
Great effort is put into finding alternatives to one-hot-encoding in order to overcome its drawbacks especially for high cardinality.
To mention a few there is the [supervised ratio or the weight of evidence](https://www.kdnuggets.com/2016/08/include-high-cardinality-attributes-predictive-model.html).
Another one worth mentioning is described in [this paper](https://kaggle2.blob.core.windows.net/forum-message-attachments/225952/7441/high%20cardinality%20categoricals.pdf), which I would prefer naming after its author Micci-Bareca.
These transformations were especially useful for the recent [Porto Seguro Kaggle Competition](https://www.kaggle.com/c/porto-seguro-safe-driver-prediction), where high-cardinality features were present.

The idea is to transform the categorical values to continuous values corresponding to their target frequency.
For example, if the categorical value **2** very often comes with the boolean target *True* or *1*, then all observations with a categorical **2** will be transformed to e.g. 0.97.

Please note, that these transformations are heavily dependent on the target distribution over the specific categorical feature.
Since we are kind of introducing target information into our dependent variables, we might run into *data leakage*, where we consider too much information from the target variable that we actually try to predict.
In the worst case we'll get over-fitting.
Again, there are several ways to mitigate that, for instance, by adding noise to the transformations.

# Custom Functions for Pandas aggregate
Now how can we apply those alternatives?
Let's consider column **D** to be our target and we want to transform the PostalCode.
For a more representative example we will distribute **D** and the PostalCode now randomly over 10000 observations.
We will apply the *weight of evidence* and the *micci-barreca* transformations.

```py
import numpy as np
import pandas as pd


def _woe(s, tp, tn):
    """Weight of evidence

    woe_i = ln(P_i/TP) - ln(N_i/TN)

    :param s: pandas groupby obj
    :param tp: total positives in full series (target prior)
    :param tn: total negatives in full series
    """
    p = s.sum()
    nom = p / tp
    den = (s.count() - p) / tn
    return np.log(nom / den)


def _micci_barreca_encode(s, tp, min_samples_leaf=1, smoothing=1):
    """Micci Barreca encoding

    This transformation outputs something between supervised ratio and target
    prior, depending on smoothing level.

    :param s: pandas groupby obj
    :param tp: total positives in full series
    """
    smoothing = \
        1 / (1 + np.exp(-(s.count() - min_samples_leaf) / smoothing))
    return tp * (1 - smoothing) + s.mean() * smoothing


if __name__ == '__main__':
    n_observations = 10000
    df = pd.DataFrame({'D': np.random.randint(2, size=n_observations),
                       'PostalCode': np.random.randint(10, size=n_observations)
                      })
    print('Original Table:')
    print(df.head(10))
    target_prior = df['D'].sum()
    target_size = df['D'].count()
    aggregation_agenda = \
        {'_woe': lambda x: _woe(x, target_prior, target_size - target_prior),
         '_micci': lambda x: _micci_barreca_encode(x, target_prior,
                                                   min_samples_leaf=100,
                                                   smoothing=10),
         }
    col = 'PostalCode'
    transformed_df = \
        df.groupby([col], as_index=False).D\
            .agg(aggregation_agenda)\
            .rename(columns={agg_key: col+agg_key for
                             agg_key in aggregation_agenda.keys()})
    print('Transformation/Mapping table:')
    print(transformed_df.head(10))

    df = df.merge(transformed_df, how='left', on=col)
    print('Merged Table:')
    print(df.head(10))
```

After creating the artificial dataset, the target's prior and size will be computed, which is necessary for both, *weight of evidence* and the *micci-barreca* encoding.
We summarize the transformations into an **aggregation_agenda**, that will be used by pandas agg function to compile the output.
Note the use of the **group_by** function, which helps us getting the target values grouped per categorical value.

The last step would be merging both tables together, that are, the original and the transformed features.
The above code will output the following:

```bash
Original Table:
   D  PostalCode
0  0           5
1  1           8
2  1           3
3  1           0
4  1           4
5  1           5
6  1           1
7  0           3
8  0           9
9  1           6

Transformation/Mapping table:
   PostalCode  PostalCode_micci  PostalCode_woe
0           0          0.507952       -0.012996
1           1          0.501496       -0.038825
2           2          0.514535        0.013348
3           3          0.515122        0.015699
4           4          0.497996       -0.052824
5           5          0.503462       -0.030960
6           6          0.509845       -0.005424
7           7          0.534010        0.091444
8           8          0.516771        0.022304
9           9          0.511263        0.000254

Merged Table:
   D  PostalCode  PostalCode_micci  PostalCode_woe
0  0           5          0.503462       -0.030960
1  1           8          0.516771        0.022304
2  1           3          0.515122        0.015699
3  1           0          0.507952       -0.012996
4  1           4          0.497996       -0.052824
5  1           5          0.503462       -0.030960
6  1           1          0.501496       -0.038825
7  0           3          0.515122        0.015699
8  0           9          0.511263        0.000254
9  1           6          0.509845       -0.005424
```

Do not forget to drop the original *PostalCode* column when working with the merged table!

# Conclusion
We have seen how Pandas can be powered to create new features from high-cardinality features among others, that go beyond simple mathematical statistics such as *mean* or *standard deviation*.
It should be clear, that one hot encoding is not always the best choice and alternative approaches are available.
Getting familiar with the concepts of **aggregate**, **group_by** and **merge** can help everyone to facilitate high-performance data analyses with just a few lines of code.

* * *
