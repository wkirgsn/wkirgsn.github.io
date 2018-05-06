---
layout: post
title: "Auto-Downsizing data"
date: 2018-02-10
comments: true
---

There is a vast amount of data published on [Kaggle Datasets](https://www.kaggle.com/datasets), that one can download and play around with for experimental purposes.
However, more than often this data is stored with data types being an overkill for the datum's intentional use.

Consider this virtual example

{% highlight bash %}
>>> import pandas as pd
>>> import numpy as np

>>> df = pd.DataFrame({'name': ['Walter', 'David', 'Jamie', 'Kendra', 'Zoey'], 
                       'age': [28, 31, 54, 44, 51]},
                      dtype=np.float32)

>>> df
    age    name
0  28.0  Walter
1  31.0   David
2  54.0   Jamie
3  44.0  Kendra
4  51.0    Zoey
{% endhighlight %}

# Problem

The used [numpy](http://www.numpy.org/) data type here, float32, takes four byte of disk space for every *age* datum stored in this mini data frame.
Yet it is plausible that a feature as 'age' would not get close to the upper limit of a single byte datum, that is, 256.

For small datasets there should be no concern but when working with Big Data and millions of entries in a database the occupied disk and RAM space might get exhausted unreasonably and unnecessarily fast.

Converting the demonstrated example is as simple as this:

{% highlight bash %}
>>> df.age = df.age.astype(np.uint8)
{% endhighlight %}

Now nobody's got time to skim through a dataset of multiple GBs just to identify the potential candidates for data type reduction.
It should come naturally to have an automatic function or class dealing with this reduction by just checking min and max values.

# Solution

The following mini class might come in handy for all reduction intentions. Please note, that reducing data types based on min and max values assumes there won't be any future data accumulated to the dataset, which could not fit in the newly converted data type.

{% highlight python %}
import numpy as np
import pandas as pd

from joblib import Parallel, delayed

__AUTHOR__ = 'Kirgsn'

class Reducer:
    """
    Class that takes a dict of increasingly big numpy datatypes to transform
    the data of a pandas dataframe to in order to save memory usage.
    """
    memory_scale_factor = 1024**2  # memory in MB

    def __init__(self, conv_table=None):
        """
        :param conv_table: dict with np.dtypes-strings as keys
        """
        if conv_table is None:
            self.conversion_table = \
                {'int': [np.int8, np.int16, np.int32, np.int64],
                 'uint': [np.uint8, np.uint16, np.uint32, np.uint64],
                 'float': [np.float16, np.float32, ]}
        else:
            self.conversion_table = conv_table

    def _type_candidates(self, k):
        for c in self.conversion_table[k]:
            i = np.iinfo(c) if 'int' in k else np.finfo(c)
            yield c, i

    def reduce(self, df, verbose=False):
        """Takes a dataframe and returns it with all data transformed to the
        smallest necessary types.

        :param df: pandas dataframe
        :param verbose: If True, outputs more information
        :return: pandas dataframe with reduced data types
        """
        ret_list = Parallel(n_jobs=-1)(delayed(self._reduce)
                                                (df[c], c, verbose) for c in
                                                df.columns)

        return pd.concat(ret_list, axis=1)

    def _reduce(self, s, colname, verbose):

        # skip NaNs
        if s.isnull().any():
            if verbose:
                print(colname, 'has NaNs - Skip..')
            return s

        # detect kind of type
        coltype = s.dtype
        if np.issubdtype(coltype, np.integer):
            conv_key = 'int' if s.min() < 0 else 'uint'
        elif np.issubdtype(coltype, np.float):
            conv_key = 'float'
        else:
            if verbose:
                print(colname, 'is', coltype, '- Skip..')
            print(colname, 'is', coltype, '- Skip..')
            return s

        # find right candidate
        for cand, cand_info in self._type_candidates(conv_key):
            if s.max() <= cand_info.max and s.min() >= cand_info.min:

                if verbose:
                    print('convert', colname, 'to', str(cand))
                return s.astype(cand)

        # reaching this code is bad. Probably there are inf, or other high numbs
        print(("WARNING: {} " 
               "doesn't fit the grid with \nmax: {} "
               "and \nmin: {}").format(colname, s.max(), s.min()))
        print('Dropping it..')
{% endhighlight %}

This class can be used the following way:

{% highlight py %}
import pandas as pd

dataset = pd.read_csv('path/to/data')
reducer = Reducer()
dataset = reducer.reduce(dataset)
{% endhighlight %}

The minimum datatypes to be converted can be controlled by the 'conv_table' that is an optional argument for the class' init function.
The default converts to all integers and at most to float32. 
Neat thing about this presented implementation: It even takes advantage of multiple CPU cores when available via the [joblib](https://pythonhosted.org/joblib/parallel.html) python library.

# Conclusion

Dealing with inappropriate data types can be now a thing of the past by utilizing sophisticated reducing classes.
Converting the dataset and saving it again to disk can save a considerable amount of disk space and, even more crucial, lessens the allocated RAM such that more actual data can be taken into account.

* * *

