# FastSHAP
> This project brings in part of the `SHAP` library into `fastai2` tabular and make it compatable. Thank you to Nestor Demeure for his assistance with the project!


## Install

`pip install fastshap`

## How to use


```
from fastshap.interp import *
```
We'll assume `learn` is a fully trained `fastai2` model.
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
