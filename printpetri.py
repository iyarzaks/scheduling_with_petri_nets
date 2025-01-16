from a_star_solver import init_real_problem
import json

def to_json():
    petri_example, rcpsp_example = init_real_problem("C:/Users/idolu/source/repos/scheduling_with_petri_nets/extract_problems/data/j30.sm.tgz/j301_1.sm",True)
    #petri_example, rcpsp_example = init_real_problem("C:/Users/User/Desktop/scheduling_with_petri_nets/extract_problems/data/j30.sm.tgz/j301_1.sm",True)

    #add all the data from rcpsp to a list
    rcpsp_total_data = [act.__dict__ for act in rcpsp_example.activities]
    rcpsp_total_data.append(rcpsp_example.backward_dependencies)
    rcpsp_total_data.append(rcpsp_example.dependencies)
    rcpsp_total_data.append(rcpsp_example.resources)

    #write all the rcpsp_example data into the json file:
    with open("rcpspExample.json", "w") as json_file:
        json.dump(rcpsp_total_data, json_file, indent=4)

    #add all data from petri to a list:
    petri_total_data = []
    petri_total_data.append(len(petri_example.places))
    for place in petri_example.places:
        petri_total_data.append(place.__dict__)


    petri_total_data.append(len(petri_example.places_dict.values()))
    for val in petri_example.places_dict.values():
        petri_total_data.append(val.__dict__)


    petri_total_data.append(len(petri_example.transitions))
    for trasition in petri_example.transitions:
        petri_total_data.append(trasition.__dict__)


    petri_total_data.append(len(petri_example.transitions_dict.values()))
    for val in petri_example.transitions_dict.values():
        petri_total_data.append(val.__dict__)



    #write all the petri_example data into the json file:
    with open("petriExample.json", "w") as json_file:
        json.dump(petri_total_data, json_file, indent=4)






if __name__=="__main__":
    to_json()