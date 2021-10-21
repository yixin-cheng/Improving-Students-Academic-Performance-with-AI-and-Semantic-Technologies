import sys
import textrazor
import pandas as pd
import networkx as nx

textrazor.api_key = "c39186d542928cf29f8514f1a57a135b124b626b571b64a74eaf28fd"

df_uniandes_descriptions = pd.read_csv("../data_Uniandes_Descriptions/ECON_Course_Content.csv", sep=";")
course_descriptions = df_uniandes_descriptions.values

client = textrazor.TextRazor(extractors=["entities", "topics"])