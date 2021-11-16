import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
file_path='result/similarity_measurement/similarity_4000_level.csv' #change the different files to get different figure of result.
data=pd.read_csv(file_path)
heat=data.pivot('course1','course2','score')
plt.figure(figsize=(15, 15))
ax=sns.heatmap(heat,cbar=1, square=1, annot_kws={'size': 15}, cmap= 'coolwarm')
plt.show()


