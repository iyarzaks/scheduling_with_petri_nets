from a_star_solver import init_real_problem
import json

def to_json():
    petri_example, rcpsp_example = init_real_problem("C:/Users/User/Desktop/scheduling_with_petri_nets/extract_problems/data/j30.sm.tgz/j301_1.sm",True)
    #add all the data from rcpsp to a list
    total_data = [act.__dict__ for act in rcpsp_example.activities]
    total_data.append(rcpsp_example.backward_dependencies)
    total_data.append(rcpsp_example.dependencies)

    #write all the data into the json file:
    with open("rcpspExample.json", "w") as json_file:
        json.dump(total_data, json_file, indent=4)

    


if __name__=="__main__":
    to_json()