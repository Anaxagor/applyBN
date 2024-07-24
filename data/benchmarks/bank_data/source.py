from ucimlrepo import fetch_ucirepo
import pandas as pd
bank_marketing = fetch_ucirepo(id=222)

X = bank_marketing.data.features.drop(["poutcome", "contact"], axis=1)
y = bank_marketing.data.targets

X = pd.concat([X, y], axis=1).dropna()

X.to_csv("bank_data.csv")
