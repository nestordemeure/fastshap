# FastSHAP
> This project brings in part of the `SHAP` library into `fastai2` and make it compatable. Thank you to Nestor Demeure for his assistance with the project!


## Install

`pip install fastshap`

## How to use

First we'll quickly train a `ADULTS` tabular model

```
from fastai2.tabular.all import *
```

```
path = untar_data(URLs.ADULT_SAMPLE)
df = pd.read_csv(path/'adult.csv')
```

```
dep_var = 'salary'
cat_names = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race']
cont_names = ['age', 'fnlwgt', 'education-num']
procs = [Categorify, FillMissing, Normalize]
```

```
splits = IndexSplitter(list(range(800,1000)))(range_of(df))
to = TabularPandas(df, procs, cat_names, cont_names, y_names="salary", splits=splits)
dls = to.dataloaders()
```

```
learn = tabular_learner(dls, layers=[200,100], metrics=accuracy)
learn.fit(1, 1e-2)
```

And now for some example usage!

```
from fastshap.interp import *
```

```
exp = ShapInterpretation(learn, df.iloc[:100])
```

```
exp.dependence_plot('age')
```


    Classification model detected, displaying score for the class <50k.
    (use `class_id` to specify another class)



![png](docs/images/output_13_2.png)


For more examples see [01_Interpret](https://muellerzr.github.io/fastshap//interpret)
