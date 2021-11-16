# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 20:04:48 2018

@author: rubenmanrique

# Ruben Manrique 2017-08-24
# Modified:  2018-01-24

"""

from SPARQLWrapper import SPARQLWrapper, JSON
import configparser
import math

class SemRefD:

    # Function Indicator
    fun_indi = " define input:default-graph-uri <http://localhost:8090/DAV>\
                 PREFIX :     <http://dbpedia.org/resource/> \
                 PREFIX dbo:     <http://dbpedia.org/ontology/> \
                 PREFIX dbp:     <http://dbpedia.org/property/> \
                 SELECT DISTINCT ?s1 FROM <http://dbpedia.org> WHERE  { \
                 ?s1 ?p1 %s  . \
                 FILTER(?p1 NOT IN (dbo:wikiPageDisambiguates, dbp:wikiPageUsesTemplate, dbo:wikiPageRedirects)) \
                 FILTER(STRSTARTS(STR(?s1), 'http://dbpedia.org/resource/')) \
                 }";

    # Function Indicator Count
    fun_indi_count = " define input:default-graph-uri <http://localhost:8090/DAV>\
                 PREFIX :     <http://dbpedia.org/resource/> \
                 PREFIX dbo:     <http://dbpedia.org/ontology/> \
                 PREFIX dbp:     <http://dbpedia.org/property/> \
                 SELECT DISTINCT COUNT(?s1) as ?count FROM <http://dbpedia.org> WHERE  { \
                 ?s1 ?p1 %s  . \
                 FILTER(?p1 NOT IN (dbo:wikiPageDisambiguates, dbp:wikiPageUsesTemplate, dbo:wikiPageRedirects)) \
                 FILTER(STRSTARTS(STR(?s1), 'http://dbpedia.org/resource/')) \
                 }";

    # Function Neighbors
    fun_neigh = " define input:default-graph-uri <http://localhost:8090/DAV>\
                 PREFIX :     <http://dbpedia.org/resource/> \
                 PREFIX dbo:     <http://dbpedia.org/ontology/> \
                 PREFIX dbp:     <http://dbpedia.org/property/> \
                 SELECT  ?p1 ?o1 FROM <http://dbpedia.org> WHERE  { \
                 %s ?p1  ?o1. \
                 FILTER(?p1 NOT IN (dbo:wikiPageDisambiguates, dbp:wikiPageUsesTemplate, dbo:wikiPageRedirects)) \
                 FILTER(STRSTARTS(STR(?o1), 'http://dbpedia.org/resource/')) \
                 }";


    # Function Resolves Redirects
    fun_redi = " define input:default-graph-uri <http://localhost:8090/DAV>\
                 PREFIX :     <http://dbpedia.org/resource/> \
                 PREFIX dbo:     <http://dbpedia.org/ontology/> \
                 PREFIX dbp:     <http://dbpedia.org/property/> \
                 SELECT  DISTINCT ?o1 FROM <http://dbpedia.org> WHERE  { \
                 %s dbo:wikiPageRedirects ?o1. \
                 FILTER(STRSTARTS(STR(?o1), 'http://dbpedia.org/resource/')) \
                 }";

    df_dict = {} #Save the df of each concept to reduce complexity

    DBpediaInstances2016 = 4678230.0

    def __init__(self, iconcepta = '', iconceptb= '', weightfunc = ''):
        """ Constructor """
        self.concepta = iconcepta
        self.conceptb = iconceptb
        self.wfunc = weightfunc
        config = configparser.ConfigParser()
        config.read('config.cfg')
        self.virtuoso = config.get('Virtuoso2', 'endpoint')
        self.indiConcepta = [] #Save the resources that point to the concept a
        self.indiConceptb = [] #Save the resources that point to the concept b

        self.neighConcepta = [] #Save the resources that point to concept a with the property through which it was retrieved
        self.neighConceptb = [] #Save the resources that point to concept b with the property through which it was retrieved
        self.capturarIndicadores()


    def capturarIndicadores(self):
        consultainidi = self.fun_indi % (self.limpiaRecursos(self.concepta))
        # print(consultainidi)
        resultoCC=self.consulta(consultainidi)
        for resul in resultoCC['results']['bindings']:
            recurso = resul['s1']['value']
            self.indiConcepta.append(recurso)


        consultaneight = self.fun_neigh % (self.limpiaRecursos(self.concepta))
        # print(consultaneight)
        resultoCC=self.consulta(consultaneight)
        for resul in resultoCC['results']['bindings']:
            recurso = resul['o1']['value']
            self.neighConcepta.append(recurso)

        consultainidi = self.fun_indi % (self.limpiaRecursos(self.conceptb))
        # print(consultainidi)
        resultoCC=self.consulta(consultainidi)
        for resul in resultoCC['results']['bindings']:
            recurso = resul['s1']['value']
            self.indiConceptb.append(recurso)

        consultaneight = self.fun_neigh % (self.limpiaRecursos(self.conceptb))
        # print(consultaneight)
        resultoCC=self.consulta(consultaneight)
        for resul in resultoCC['results']['bindings']:
            recurso = resul['o1']['value']
            self.neighConceptb.append(recurso)

        #After recovering the redirects there may be duplicates in INDIS
        self.indiConcepta = list(set(self.indiConcepta))
        self.indiConceptb = list(set(self.indiConceptb))

        if not self.indiConcepta:
            raise ValueError("Error: Indi list vacia para concepto A")

        if not self.indiConceptb:
            raise ValueError("Error: Indi list vacia para concepto B")

        if not self.neighConcepta:
            raise ValueError("Error: Neigh list vacia para concepto A")

        if not self.neighConceptb:
            raise ValueError("Error: Neigh list vacia para concepto B")


    def calculaRefD(self):
        sumaN1 = 0.0
        for concept in self.indiConceptb:
            sumaN1 += self.Wfunctions(concept,self.concepta)

        sumaN2 = 0.0
        for concept in self.indiConcepta:
            sumaN2 += self.Wfunctions(concept,self.conceptb)

        sumaD1 = 0.0
        for concept in list(set(self.neighConcepta)):
            sumaD1 += self.Wfunctions(concept,self.concepta)

        sumaD2 = 0.0
        for concept in list(set(self.neighConceptb)):
            sumaD2 += self.Wfunctions(concept,self.conceptb)

        return ((sumaN1/sumaD1) - (sumaN2/sumaD2))

    def Wequals(self,conceptC,ConceptAB ):
        if ConceptAB == self.concepta:
            if conceptC in self.neighConcepta:
                return 1
            else:
                return 0
        elif ConceptAB == self.conceptb:
            if conceptC in self.neighConceptb:
                return 1
            else:
                return 0
        else:
            return None

    def Wfunctions(self,conceptC,ConceptAB):
        if conceptC.replace("http://dbpedia.org/resource","").strip() == ConceptAB:
            return 0.0

        if self.wfunc == 'equals':
            return self.Wequals(conceptC,ConceptAB)
        elif self.wfunc == 'tfidf':
            return self.Wtfidf(conceptC,ConceptAB)
        else:
            raise ValueError('Funcion de ponderacion no definida')


    def Wtfidf(self, conceptC, ConceptAB):

        if ConceptAB == self.concepta:
            tf = self.neighConcepta.count(conceptC)
        elif ConceptAB == self.conceptb:
            tf = self.neighConceptb.count(conceptC)
        else:
            print("Error, revisar Wtfidf")
            return None

        if tf == 0:
            return 0.0

        if conceptC not in self.df_dict:
            consultadf = self.fun_indi_count % (self.limpiaRecursos(conceptC))
            resultoCC=self.consulta(consultadf)
            for resul in resultoCC['results']['bindings']:
                df = float(resul['count']['value']) + 1 # The number of Wikipedia articles where the concept C appears= Los recursos que tienen un enlace al concepto C mas el articulo del concepto mismo
            self.df_dict[conceptC] = df
        else:
            df = self.df_dict[conceptC]

        return tf * math.log(self.DBpediaInstances2016/df)



    def consulta(self, sqlQuery):
        """Run query"""
        #print(sqlQuery)
        sparql = SPARQLWrapper(self.virtuoso)
        sparql.setCredentials(user="dba", passwd = "dba")
        sparql.setReturnFormat(JSON)
        sparql.setQuery(sqlQuery)
        results = sparql.query()
        results = results.convert()
        # print(results)
        return results

    def limpiaRecursos(self, recursoDirty):
        """Clean from () queries SPARQL"""
        recursoDirty = recursoDirty.replace(" ","_")
        if "http://dbpedia.org/resource" not in recursoDirty:
            recursoDirty = "http://dbpedia.org/resource/" + recursoDirty
        recursoClean = "<" + recursoDirty + ">"
        return recursoClean
