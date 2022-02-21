from sklearn.preprocessing import LabelBinarizer


lb = LabelBinarizer()
print(lb.fit_transform(['yes', 'inland', 'near bay', 'near_ocean', 'yes']))
