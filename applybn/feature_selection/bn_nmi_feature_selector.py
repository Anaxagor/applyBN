from pyitlib import discrete_random_variable as drv
from sklearn.feature_selection import SelectorMixin
from sklearn.base import BaseEstimator
import pandas as pd
import numpy as np
import logging


class NMIFeatureSelector(BaseEstimator, SelectorMixin):

  def __init__(self, threshhold = 0.5, bin_count=100, verbose=False):
    '''
    This class performs feature selection based on Normalized Mutual Information (NMI).
    
    Args:
        threshold (float): The minimum NMI to include a feature in the selection.
        bin_count (int): Number of bins for float discretization.
        verbose (bool): If True, logging intermediate information.
    '''
    self.threshhold = threshhold
    self.bin_count = bin_count
    self.bin_labels = [i for i in range(self.bin_count)]
    self.verbose = verbose
    self.entropy_map = dict()


  def _discreticise(self, x, y=None):
    '''
    Prepare data to feed into pyitlib drv.entropy
    Calculate happenings of each event in descreticised feature
    '''
    arr_x = np.array(x, dtype=int)
    
    if y is None:
      # if only one given, return bincount integers from 0 to max(x)
      return np.bincount(arr_x, minlength=max(1, max(x) - min(x) + 1)).tolist()
    else:
      arr_y = np.array(y, dtype=int)
      x_min, x_max = arr_x.min(), arr_x.max()
      y_min, y_max = arr_y.min(), arr_y.max()
      
      hist, _, _ = np.histogram2d(
        arr_x, arr_y,
        bins=[x_max - x_min + 1, y_max - y_min + 1],
        range=[[x_min, x_max + 1], [y_min, y_max + 1]]
      )
      return hist.flatten().astype(int).tolist()


  def _normalized_mutual_information(self, a, b):
    '''
    Calculate Normalized Mutual Information (NMI) for pair of features

    Args: 
        a, b - pair of pd.Series to calculate NMI.
    '''
    if (a.name, None) not in self.entropy_map:
      self.entropy_map[(a.name, None)] = float(drv.entropy(self._discreticise(a)))
    if (b.name, None) not in self.entropy_map:
      self.entropy_map[(b.name, None)] = float(drv.entropy(self._discreticise(b)))
    if (a.name, b.name) not in self.entropy_map:
      self.entropy_map[(a.name, b.name)] = float(drv.entropy_joint(self._discreticise(a,b)))

    Ha = self.entropy_map[(a.name, None)]
    Hb = self.entropy_map[(b.name, None)]
    Hab = self.entropy_map[(a.name, b.name)]
    return (Ha + Hb - Hab) / (2 * max(Ha, Hb))

  def _prepare_dataset(self, df: pd.DataFrame):
    '''
    Prepare the dataset, discretize the fields, remove rows
    Return a ready-to-use dataset
    
    Args: 
        df - data frame to be cleaned.
    '''
    def transform_column(col: pd.Series) -> pd.Series:
      # Handle integer-type columns by enumerating unique values
      if pd.api.types.is_integer_dtype(col.dtype):
          mapping = {val: i for i, val in enumerate(col.unique())}
          return col.map(mapping)
      # Handle numeric columns (floats) using binning
      elif pd.api.types.is_numeric_dtype(col.dtype):
          binned = pd.cut(col, bins=self.bin_count, labels=self.bin_labels, duplicates='drop')
          # Add an extra category for any values outside the bin ranges
          return binned.cat.add_categories(self.bin_count + 1).fillna(self.bin_count + 1)
      # Handle other column types (e.g., strings)
      else:
          mapping = {val: i for i, val in enumerate(col.unique())}
          return col.map(mapping)
    # Apply the transform_column function to each column in the DataFrame
    clean_df = df.apply(transform_column)
    return clean_df

  def fit(self, X: pd.DataFrame, y: pd.Series = None):
    if y is None:
      return self
    
    if X.shape[0] != y.shape[0]:
      raise ValueError

    df = X.copy()
    df[y.name] = y
    target = y.name
    self.initial_features = list(X.columns)

    clean_df = self._prepare_dataset(df)
    if self.verbose:
      logging.info('Dataset is ready')

    nmi_map = dict()
    self.entropy_map = dict()
    pre_features = []
    def first_selection(col: pd.Series) -> pd.Series:
      if col.name != target:
        if (col.name, target) not in nmi_map.keys():
          nmi_map[(col.name, target)] = self._normalized_mutual_information(col, clean_df[target])
        nmi = nmi_map[(col.name, target)]
        if nmi > self.threshhold:
          pre_features.append(col.name)
      return col

    clean_df.apply(first_selection)

    if self.verbose:
      logging.info(f'First selection done, {len(pre_features)} selected', pre_features)

    if len(pre_features) == 1:
      self.features_selected = pre_features
      self.feature_names_in_ = X.columns
      self.n_features_in_ = X.shape[1]
      self.is_fitted_ = True
      self.X_ = X
      self.y_ = y
      return self

    features = []
    def second_selection(col: pd.Series) -> pd.Series:
      if col1 != col.name and col1 != target and col.name != target:
        if (col1, col.name) not in nmi_map:
          nmi_map[(col1, col.name)] = self._normalized_mutual_information(clean_df[col1], col)

        nmi_tg_1 = nmi_map[(col1, target)]
        nmi_tg_2 = nmi_map[(col.name, target)]
        nmi_col_12 = nmi_map[(col1, col.name)]

        if (nmi_tg_1 > nmi_tg_2) and (nmi_col_12 > nmi_tg_2):
          if col1 not in features:
            features.append(col1)
      return col

    for col1 in pre_features:
      clean_df.apply(second_selection)
    
    if self.verbose:
      logging.info(f'Second selection done, reduced: {len(pre_features)} -> {len(features)}')

    self.entropy_map.clear()
    self.features_selected = features
    self.feature_names_in_ = X.columns
    self.n_features_in_ = X.shape[1]
    self.is_fitted_ = True
    self.X_ = X
    self.y_ = y

    return self

  def _get_support_mask(self):
    mask = [True if e in self.features_selected else False for i, e in enumerate(self.initial_features)]
    return np.array(mask)