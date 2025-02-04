# Feature Selection

## BNFeatureSelector

::: applybn.feature_selection.bn_feature_selector.BNFeatureSelector

## CausalFeatureSelector

::: applybn.feature_selection.ce_feature_selector.CausalFeatureSelector

```python
import pandas as pd
from applybn.feature_selection.ce_feature_selector import CausalFeatureSelector

df = pd.read_csv('datat.csv') 
target = df['target']

features = df.drop(columns=['target'])
selector = CausalFeatureSelector(n_bins='auto')
selector.fit(features.values, target.values)

print(selector.transform(features.values))
```
