"""
This file is for the data pre-processing
"""
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder


class Preprocessor:
        def __init__(self,path,new_path,subset):
            self.subset=subset
            self.path=path
            self.new_path=new_path
            self.dtype = {'ano_curriculo': np.string_, 'cod_curriculo': np.string_, 'mat_ano': np.int8,
                          'mat_sem': np.int8,
                          'periodo': np.string_,
                          'ano': np.string_, 'semestre': np.int8, 'semestre_recomendado': np.int8,
                          'semestre_do_aluno': np.int8,
                          'no_creditos': np.int8,
                          'cep': np.string_, 'puntos_enem': np.float32, 'diff': np.int8, 'tentativas': np.int8,
                          'cant': np.int8,
                          'identificador': np.string_, 'cod_curriculo': np.int8, 'cod_enfase': np.string_}
            self.dfh = pd.read_csv(self.path, sep=';', dtype=self.dtype,
                                   converters={'grau': lambda x: x.replace(',', '.')})

            self.organise()

        def subset_seperator(self):
            """
            This is for separating raw dataset by degree
            """
            self.dfh= self.dfh[self.dfh['cod_curso'] == self.subset]
            print('Subset %s is being selected' % self.subset)

        def organise(self):
            """
            This function is for data cleaning, data normalization and Data validation.
            :return:
            """
            self.subset_seperator()
            self.dfh = self.dfh.applymap(lambda x: x.strip() if type(
                x) is str else x)  # remove the beginning and end space of string, apply it in each element

            # remove all null and redundant columns
            drop_list=[0,1,2,3,4,5,14,29]
            self.dfh = self.dfh.drop(self.dfh.columns[drop_list],axis=1)
            self.dfh = self.dfh.sort_values(['matricula', 'semestre_do_aluno'])
            precessing_list=[2,8,12,14,16,17,22,23]
            for i in precessing_list:
                self.dfh.iloc[:, i] = LabelEncoder().fit_transform(self.dfh.iloc[:, i])

            self.dfh['sit_vinculo_atual']=self.dfh['sit_vinculo_atual'].replace(['DESLIGADO','MATRICULA EM ABANDONO','JUBILADO'],0)

            self.dfh['sit_vinculo_atual'] = self.dfh['sit_vinculo_atual'].replace(['FORMADO','MATRICULADO','MATRICULADO','HABILITADO A FORMATURA',
                                                                                   'MATRICULA DESATIVADA','MATRICULA TRANCADA','TRANSFERIDO PARA OUTRA IES',
                                                                                   'EM ADMISSAO','MATRICULADO EM CONVENIO','FALECIDO'], 1)
            self.dfh.to_csv(self.new_path, index=False)
            self.dfh = self.dfh.apply(pd.to_numeric)
            self.dfh.to_csv(self.new_path, header=True, index=False)
            # data imputation
            self.dfh=self.impute_missing_value()
            # move the dropout status to the second end
            column_to_move_grades = self.dfh.pop("grau")
            self.dfh["grau"] = column_to_move_grades
            # move the dropout status to the end
            column_to_move = self.dfh.pop("sit_vinculo_atual")
            self.dfh["sit_vinculo_atual"] = column_to_move
            # normalise input data
            for column in self.dfh.columns[:-1]:
                # the last column is target
                self.dfh[column] = self.dfh.loc[:, [column]].apply(lambda x: (x - x.mean()) / x.std())
            print('End of data-preprocessing')
            self.dfh=self.dfh.dropna(axis=1)
            self.dfh.to_csv(self.new_path,header=None,index=False)

        def impute_missing_value(self):
            """
            this function written for handling missing values wtih using random forest
            the order of imputation is from the least column to the most ones first, fill the missing values with 0 in other columns
            then run the algorithm and do the iteration.
            """
            print("Getting into data imputation")
            df = pd.read_csv(self.new_path)
            y_full = pd.read_csv(self.new_path)
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

            return X_missing_reg






