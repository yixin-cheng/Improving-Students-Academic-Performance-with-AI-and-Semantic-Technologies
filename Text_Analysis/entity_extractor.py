import sys
import textrazor
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import RefDSimply as rf

textrazor.api_key = "c39186d542928cf29f8514f1a57a135b124b626b571b64a74eaf28fd"

df_descriptions = pd.read_csv("raw_data_csv/course_description.csv")
course_descriptions = df_descriptions.values

print(course_descriptions)
client = textrazor.TextRazor(extractors=["entities", "topics"])
textrazor_response_dict = {}
for row in  course_descriptions:
    # row[1] = row[1].replace("(;;;)", ",")
    textrazor_response_dict[row[0]] = client.analyze(row[1])
    print(str(row[0])+" "+str(len(textrazor_response_dict[row[0]].entities())))
clean_textrazor_response_dict = {}
for course, description in textrazor_response_dict.items():
    current_course_concept_list = []
    for entity in description.entities():
        if entity.english_id:
            if not entity.english_id in current_course_concept_list:
                current_course_concept_list.append(entity.english_id)
    clean_textrazor_response_dict[course] = current_course_concept_list

for course, concept_list in clean_textrazor_response_dict.items():
    print(course)
    print(concept_list)

result={}

def compare_concept_lists(concept_list, concept_list2):
    for concept_a in concept_list:
        for concept_b in concept_list2:
            mmss = rf.SemRefD(concept_a, concept_b, "equals")
            refdE = mmss.calculaRefD()
            result[concept_a+':'+concept_b]=refdE
            print(concept_a + " - " + concept_b + " : " + str(refdE))

compare_concept_lists(list(clean_textrazor_response_dict.values())[0],list(clean_textrazor_response_dict.values())[1])

# print(result)

data_items=result.items()
data_list=list(data_items)

df=pd.DataFrame(data_list)

df.to_csv("Ref_result_1110_2100.csv",index=False)





# course_concept_association_G = nx.Graph()
#
# for course, concept_list in clean_textrazor_response_dict.items():
#     course_concept_association_G.add_node(course)
#
# for course, concept_list in clean_textrazor_response_dict.items():
#     for course2, concept_list2 in clean_textrazor_response_dict.items():
#         number_of_connections = compare_concept_lists(concept_list, concept_list2)
#         if number_of_connections > 0:
#             for concept in concept_list:
#                 if concept in concept_list2 and course != course2:
#                     course_concept_association_G.add_edge(course, course2, weight=number_of_connections)
# fig, ax = plt.subplots(1, 1, figsize=(15, 15));
# ax.pcolor = "red"
# nx.draw_networkx(course_concept_association_G, ax=ax)