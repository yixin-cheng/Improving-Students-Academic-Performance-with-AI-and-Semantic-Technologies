from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
list=['comp1100','comp1110','comp1600','comp1710','comp1730','comp2100','comp2120',
          'comp2310','comp2400','comp2410','comp2420','comp2550','comp2560','comp2610',
          'comp2620','comp2700','comp3120','comp3300','comp3310','comp3320','comp3425',
          'comp3430','comp3530','comp3600','comp3620','comp3701','comp3702','comp3703',
          'comp3704','comp3900','comp4300','comp4330','comp4610','comp4620','comp4670',
          'comp4691']
df1=pd.DataFrame(columns=['1_course','2_course','Similarity'])
for i in range(len(list)):
    for j in range(len(list)):
        list1 = []
        df = pd.read_csv('full/'+list[i]+'_'+list[j]+'.csv')
        col_list = [df['Similarity'].to_list()]
        mean=np.mean(col_list[0])
        print(np.mean(col_list[0]))
        # df1 = pd.read_csv('a.csv')
        list1.append(list[i])
        list1.append(list[j])
        list1.append(mean)
        indexsize=df1.index.size
        df1.loc[indexsize]=list1

        df1.to_csv('result_new.csv',float_format='%.4f',index=None)









# list=[[1]*len(col_list[0])]
# for i in range (len(col_list[0])):
#     list[0].append(1)

# print(list)

# sim=cosine_similarity(col_list, list)

# print(sim)




