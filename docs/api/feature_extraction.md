# Feature Extraction


## BNFeatureGenerator
::: applybn.feature_extraction.bn_feature_extractor.BNFeatureGenerator

### Example

```python
data = pd.read_csv('your_data.csv')
generator = BNFeatureGenerator()
generator.fit(data)
new_features = generator.transform(data)
data_combined = pd.concat([data, new_features], axis=1)
```
