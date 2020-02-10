# FastSHAP
> This project brings in part of the `SHAP` library into `fastai2` and make it compatable. Currently just supports the `tabular` module for the following plots:


## Install

`pip install fastshap`

## How to use

```
exp = KernelExplainer(learn, df.iloc[:100])
```

```
exp.dependance_plot('age')
```

```
exp.decision_plot()
```
