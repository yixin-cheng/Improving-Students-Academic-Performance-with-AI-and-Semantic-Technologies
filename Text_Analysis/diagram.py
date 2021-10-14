import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
data=pd.read_csv("result_new_full.csv")
list=[]
for i in range (len(data)):
    if data.iloc[i,0]==data.iloc[i,1]:
        data.iloc[i,2]=0.92
    if data.iloc[i,0]=='comp3702' or data.iloc[i,1]=='comp3702'or \
            data.iloc[i,0]=='comp4880' or data.iloc[i,1]=='comp4880':
        list.append(i)
data=data.drop(list)
data.to_csv("result_new_full.csv",index=False)


heat=data.pivot('s1','s2','score')
plt.figure(figsize=(15, 15))
ax=sns.heatmap(heat,cbar=1, square=1, annot_kws={'size': 15}, cmap= 'coolwarm')
# heat=data.pivot('1_Course','2_Course','Similarity')
# ax=sns.heatmap(heat)
ax.figure.savefig("output.png")
# plt.savefig('result.png')
plt.show()
# plt.savefig('result.png')

