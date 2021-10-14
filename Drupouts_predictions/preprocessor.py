import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
import re
from sklearn.preprocessing import LabelEncoder

class Preprocessor:
        def __init__(self,path,new_path):
            self.path=path
            self.new_path=new_path




        def organise(self):
            self.dtype = {'ano_curriculo': np.string_, 'cod_curriculo': np.string_, 'mat_ano': np.int8, 'mat_sem': np.int8,
                     'periodo': np.string_,
                     'ano': np.string_, 'semestre': np.int8, 'semestre_recomendado': np.int8,
                     'semestre_do_aluno': np.int8,
                     'no_creditos': np.int8,
                     'cep': np.string_, 'puntos_enem': np.float32, 'diff': np.int8, 'tentativas': np.int8,
                     'cant': np.int8,
                     'identificador': np.string_, 'cod_curriculo': np.int8, 'cod_enfase': np.string_}
            self.dfh = pd.read_csv(self.path, sep=';', dtype=self.dtype,
                              converters={'grau': lambda x: x.replace(',', '.')})
            self.dfh = self.dfh.applymap(lambda x: x.strip() if type(
                x) is str else x)  # remove the beginning and end space of string, apply it in each element
            self.dfh=self.dfh.drop(['cod_enfase','count'],axis=1)
            precessing_list=[0,1,2,3,4,5,7,10,12,13,14,18,20,22,28,29]
            for i in precessing_list:
                self.dfh.iloc[:, i] = LabelEncoder().fit_transform(self.dfh.iloc[:, i])
            for i in range(len(self.dfh['sit_vinculo_atual'])):
                if self.dfh['sit_vinculo_atual'][i] == ('DESLIGADO' or 'JUBILADO' or 'MATRICULA EM ABANDONO'):
                    self.dfh.loc[i, 'sit_vinculo_atual'] = 0
                else:
                    self.dfh.loc[i, 'sit_vinculo_atual'] = 1
            self.dfh = self.dfh.apply(pd.to_numeric)
            for column in self.dfh.columns[:-1]:
                # the last column is target
                self.dfh[column] = self.dfh.loc[:, [column]].apply(lambda x: x / x.max())
            self.dfh.to_csv('222.csv')
            return self.dfh

        def impute_missing_value(self):
            """
            this function written for handling missing values wtih using random forest
            the order of imputation is from the least column to the most ones first, fill the missing values with 0 in other columns
            then run the algorithm and do the iteration.
            """
            df = self.organise()
            print(df)
            y_full = self.organise()
            X_missing_reg = df.copy()  # temp df
            # get columns containing nan
            nan_values = df.isna()
            nan_columns = nan_values.any()

            columns_with_nan = df.columns[nan_columns].tolist()
            #get the index of the nan column
            new_columns_with_nan=[]
            for i in columns_with_nan:
                index_no = df.columns.get_loc(i)
                new_columns_with_nan.append(index_no)
            # print(new_columns_with_nan)
            for i in new_columns_with_nan:
                df = X_missing_reg
                # construct new pattern matrix
                fillc = df.iloc[:, i]  # all rows
                df = pd.concat([df.iloc[:, df.columns != i], pd.DataFrame(y_full)], axis=1)
                # fill 0 in all cells that have missing values
                df_0 = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0).fit_transform(df)
                Ytrain = fillc[fillc.notnull()]
                Ytest = fillc[fillc.isnull()]
                Xtrain = df_0[Ytrain.index, :]
                Xtest = df_0[Ytest.index, :]
                # use random forest to impute missing value
                rfc = RandomForestRegressor(n_estimators=100)
                rfc = rfc.fit(Xtrain, Ytrain)
                Ypredict = rfc.predict(Xtest)
                Ypredict = [round(element, 4) for element in Ypredict]
                X_missing_reg.loc[df.iloc[:, i].isnull(), df.columns[i]] = Ypredict  # fill new missing values in the cells

            X_missing_reg.to_csv(self.new_path, header=False, index=True)  # remove header in the end






