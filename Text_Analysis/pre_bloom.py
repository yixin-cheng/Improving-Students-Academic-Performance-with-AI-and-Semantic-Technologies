import pandas as pd
data=pd.read_csv('result_new_full_top1_norepeat.csv')
threshold_1=0.9
threshold_2=0.95
delete_list=[]
result=[]
new_set=set()
bloom=['remember', 'understand','apply','analyze','evaluate','create']
list1 = ['comp1100', 'comp1110', 'comp1600', 'comp1710', 'comp1730', 'comp2100', 'comp2120',
            'comp2310', 'comp2400', 'comp2410', 'comp2420', 'comp2550', 'comp2560', 'comp2610',
            'comp2620', 'comp2700', 'comp3120', 'comp3300', 'comp3310', 'comp3320', 'comp3425',
            'comp3430', 'comp3530', 'comp3600', 'comp3620', 'comp3701', 'comp3702','comp3704',
            'comp3900', 'comp4300', 'comp4330', 'comp4610', 'comp4620', 'comp4670','comp4691']
# remove records lower than threshold 1
for i in range (len(data)):
    if data.iloc[i,2]<threshold_1:
        delete_list.append(i)
data=data.drop(delete_list)
data.to_csv('final_result.csv', index=False)
# generate a list of current courses
list2=[]
for i in range(len(data)):
    if data.iloc[i,0] not in list2:
        list2.append(data.iloc[i,0])
    if data.iloc[i, 1] not in list2:
        list2.append(data.iloc[i,1])

list2=sorted(list2)
print(len(list2))
#
for i in range(len(list2)):
    for j in range(i+1,len(list2)):
        df = pd.read_csv('full_3/' + list2[i] + '_' + list2[j] + '.csv')
        # df=df.drop([0])
        df = df.iloc[:, :].apply(lambda x: x.astype(str).str.lower())
        x=0
        y=0
        print('full_3/' + list2[i] + '_' + list2[j] + '.csv')
        # print(df)
        for k in range (len(df)):
            #and (m in df.iloc([k,1]) and m in df.iloc([k,2])for m in bloom)
            if float(df.iloc[k,0])>=threshold_2 and (m in df.iloc([k,1]) and m in df.iloc([k,2])for m in bloom):

                if bloom[0] in df.iloc[k,1]:
                    # print('b1')
                    x+=1
                if bloom[1] in df.iloc[k,1]:
                    # print('b2')
                    x+=2
                if bloom[2] in df.iloc[k,1]:
                    # print('b3')
                    x+=3
                if bloom[3] in df.iloc[k,1]:
                    # print('b4')
                    x+=10
                if bloom[4] in df.iloc[k,1]:
                    # print('b5')
                    x+=11
                if bloom[5] in df.iloc[k,1]:
                    # print('b6')
                    x+=12
                if bloom[0] in df.iloc[k,2]:
                    # print('1')
                    y+=1
                if bloom[1] in df.iloc[k,2]:
                    # print('1')
                    y+=2
                if bloom[2] in df.iloc[k,2]:
                    # print('1')
                    y+=3
                if bloom[3] in df.iloc[k,2]:
                    # print('4')
                    y+=10
                if bloom[4] in df.iloc[k,2]:
                    # print('3')
                    y+=11
                if bloom[5] in df.iloc[k,2]:
                    # print('2')
                    y+=12
        print('x:'+str(x))
        print('y:'+str(y))
        if x>y:
            result.append(list2[i]+'>'+list2[j])
        if x<y:
            result.append(list2[j] + '>' + list2[i])

print(result)




