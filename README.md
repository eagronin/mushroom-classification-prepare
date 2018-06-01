# Data Preparation

This section will address the issue of missing values, convert the categorical features to dummies 
and then reduce the dimensionality of the feature sapace to two principal compoenents.
Missing values in the mushroom dataset are identified as '?'.  Only one variable (stalk-root) appears to contain
missing values.  The following code outputs summary statistics for the imported data:

```python
# Read data
df = read_mushroom_data()

# Summary statistics
print('Number of samples: ', df.shape[0])
print('Number of attributes: ', df.shape[1])

value_counts = df['class'].value_counts()
e = value_counts['e']
p = value_counts['p']

print('\nEdible:    ', e)
print('Poisonous: ', p)
print('\nTotal:     ', e + p)

print('\nValues classified as \'Missing\' for stalk-root: ', (df.iloc[:,11] == '?').sum())
```

The resulting summary statistics are presented below:

```
Number of samples:  8124
Number of attributes:  23

Edible:     4208
Poisonous:  3916

Total:      8124

Values classified as 'Missing' for stalk-root:  2480
```

Due to a large number of missing values in stalk-root, this feature has been removed.
The rest of the features and the target have been converted to dummies taking values of 0 and 1.  The following code 
outputs summary statistics after applying the transformations discussed above:

```python
df = df.drop(str(df.columns[11]), axis = 1)
df2 = pd.get_dummies(df)

# Remove uninformative features (either all values are 1, or all values are 0)
df2 = df2[(df2.columns[df2.mean() != 1]) & (df2.columns[df2.mean() != 0])]

# Output summary statistics after converting the features to dummies

print('\nNumber of samples: ', df2.shape[0])
print('Number of attributes: ', df2.shape[1])

print('\nRemaining missing values across all attributes and samples: ', df2.isnull().sum().sum())

print('\nMinimum value across all attributes and samples: ', df2.min().min())
print('Maximum value across all attributes and samples: ', df2.max().max())

print('\nMinimum fraction of \'1\'-s across all attributes: {:.5f}'.format(df2.mean().min()))
print('Maximum fraction of \'1\'-s across all attributes: {:.5f}'.format(df2.mean().max()))
```

The summary statistics of this transformed dataset are presented below:

```
Number of samples:  8124
Number of attributes:  113

Remaining missing values across all attributes and samples:  0

Minimum value across all attributes and samples:  0
Maximum value across all attributes and samples:  1

Minimum fraction of '1'-s across all attributes: 0.00049
Maximum fraction of '1'-s across all attributes: 0.97538
```
