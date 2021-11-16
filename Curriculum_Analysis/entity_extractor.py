import textrazor
import pandas as pd
import RefDSimply as rf

textrazor.api_key = "c39186d542928cf29f8514f1a57a135b124b626b571b64a74eaf28fd" # textrzor key

df_descriptions = pd.read_csv("raw_data_csv/course_description.csv") # get dataset
course_descriptions = df_descriptions.values
course_list=["comp1100","comp1110","comp1600","comp2100","comp2120","comp2300","comp2310","comp2420",
             "comp3600","math1005"]
print(course_descriptions)

client = textrazor.TextRazor(extractors=["entities", "topics"]) #using textrzor
#using textrazor to extract entities
textrazor_response_dict = {}
for row in  course_descriptions:
    textrazor_response_dict[row[0]] = client.analyze(row[1])
    print(str(row[0])+" "+str(len(textrazor_response_dict[row[0]].entities())))

# get the list of entities for each course
clean_textrazor_response_dict = {}
for course, description in textrazor_response_dict.items():
    current_course_concept_list = []
    for entity in description.entities():
        if entity.english_id:
            if not entity.english_id in current_course_concept_list:
                current_course_concept_list.append(entity.english_id)
    clean_textrazor_response_dict[course] = current_course_concept_list

# compare concepts between two course in the list
def compare_concept_lists(concept_list, concept_list2,result1):
    for concept_a in concept_list:
        for concept_b in concept_list2:
            mmss = rf.SemRefD(concept_a, concept_b, "equals")
            refdE = mmss.calculaRefD()
            result1[concept_a+':'+concept_b]=refdE
            print(concept_a + " - " + concept_b + " : " + str(refdE))
# process of comparison
for index in range (len(course_list)):
    for index1 in range(index+1,len(course_list)):

        result = {}

        concept_list = list(clean_textrazor_response_dict[course_list[index]])
        concept_list2 = list(clean_textrazor_response_dict[course_list[index1]])
        print(course_list[index],concept_list)
        print(course_list[index1],concept_list2)
        compare_concept_lists(concept_list,concept_list2,result)
        data_items = result.items()
        data_list = list(data_items)

        df = pd.DataFrame(data_list)


        df.to_csv("Ref_result_"+course_list[index]+"_"+course_list[index1]+".csv", index=False)