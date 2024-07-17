import pandas as pd

# Veriyi yükle
melbourne_file_path = 'train.csv'
home_data = pd.read_csv(melbourne_file_path)

# Eksik değerleri kontrol et
#print(home_data.isnull().sum())

# Sayısal verilerdeki eksik değerleri medyan ile doldur
for col in home_data.select_dtypes(include=['float64', 'int64']).columns:
    home_data[col].fillna(home_data[col].median(), inplace=True)

# Kategorik verilerdeki eksik değerleri mod ile doldur
for col in home_data.select_dtypes(include=['object']).columns:
    home_data[col].fillna(home_data[col].mode()[0], inplace=True)

# Özelliklerin varlığını kontrol et
melbourne_features = ['LotArea', 'YearBuilt', 'Street', 'Utilities',
                      'KitchenQual', 'Heating', 'FullBath']
missing_features = [f for f in melbourne_features if f not in home_data.columns]
if missing_features:
    raise ValueError(f"Eksik özellikler: {missing_features}")

# Özellikleri seç
X = home_data[melbourne_features]

# Kategorik verileri sayısal verilere dönüştür
X = pd.get_dummies(X, columns=['Street', 'Utilities', 'KitchenQual', 'Heating'])

#print(X.head())
y = home_data.SalePrice
from sklearn.model_selection import train_test_split

# split data into training and validation data, for both features and target
# The split is based on a random number generator. Supplying a numeric value to
# the random_state argument guarantees we get the same split every time we
# run this script.
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(train_X, train_y)
melb_preds = forest_model.predict(X)

df = pd.DataFrame({'SalePrice': melb_preds})
df['ID'] = df.index

# Sütun sıralamasını güncelle: "ID" sütunu başta olacak
df = df[['ID', 'SalePrice']]
print(df)

df.to_csv('file1.csv', index=False)