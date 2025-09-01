from a_star_solver import init_real_problem
import json
import os


from pathlib import Path
from pathlib import Path
def to_json():
    """
    Process all .sm files in the dataset and convert them to JSON format.
    Saves full RCPSP data including deep dependencies, and full Petri net structure.
    """
    base_path = "extract_problems/data/j30.sm.tgz"
    output_base_dir = "json_outputs_j30"

    os.makedirs(output_base_dir, exist_ok=True)

    for group in range(1, 49):  # 48 groups
        for index in range(1, 11):  # 10 files per group
            folder_name = f"j30{group}_{index}"
            file_name = f"{folder_name}.sm"
            sm_file_path = os.path.join(base_path, file_name)
            output_folder = os.path.join(output_base_dir, folder_name)

            os.makedirs(output_folder, exist_ok=True)

            try:
                # Your parsing function
                petri_example, rcpsp_example = init_real_problem(sm_file_path, True)

                # Process RCPSP data
                rcpsp_total_data = [act.__dict__ for act in rcpsp_example.activities]
                rcpsp_total_data.append(rcpsp_example.backward_dependencies)
                rcpsp_total_data.append(rcpsp_example.dependencies)
                rcpsp_total_data.append(list(rcpsp_example.dependencies_deep_set))
                rcpsp_total_data.append(rcpsp_example.resources)

                with open(os.path.join(output_folder, "rcpsp.json"), "w") as json_file:
                    json.dump(rcpsp_total_data, json_file, indent=4)

                # Process Petri net data
                petri_total_data = []

                petri_total_data.append(len(petri_example.places))
                for place in petri_example.places:
                    petri_total_data.append(place.__dict__)

                petri_total_data.append(len(petri_example.places_dict))
                for val in petri_example.places_dict.values():
                    petri_total_data.append(val.__dict__)

                petri_total_data.append(len(petri_example.transitions))
                for transition in petri_example.transitions:
                    petri_total_data.append(transition.__dict__)

                petri_total_data.append(len(petri_example.transitions_dict))
                for val in petri_example.transitions_dict.values():
                    petri_total_data.append(val.__dict__)

                with open(os.path.join(output_folder, "petri.json"), "w") as json_file:
                    json.dump(petri_total_data, json_file, indent=4)

                print(f"Successfully processed {folder_name}")

            except Exception as e:
                print(f"Error processing {folder_name}: {str(e)}")
                # """
    # Process a single .sm file and convert it to JSON format

    # Args:
    #     file_path (str): Path to the specific .sm file to process
    # """
    # import os
    # import json
    # file_path="extract_problems/data/j30.sm.tgz/j301_1.sm"
    # output_base_dir = "temp_out"

    # # Make sure the base output directory exists
    # os.makedirs(output_base_dir, exist_ok=True)

    # # Extract filename without extension for folder naming
    # file_name = os.path.basename(file_path)
    # folder_name = os.path.splitext(file_name)[0]  # Remove .sm extension

    # output_folder = os.path.join(output_base_dir, folder_name)

    # # Create a specific folder for this file
    # os.makedirs(output_folder, exist_ok=True)

    # try:
    #     # Process the file
    #     petri_example, rcpsp_example = init_real_problem(file_path, True)

    #     # Process RCPSP data
    #     rcpsp_total_data = [act.__dict__ for act in rcpsp_example.activities]
    #     rcpsp_total_data.append(rcpsp_example.backward_dependencies)
    #     rcpsp_total_data.append(rcpsp_example.dependencies)
    #     rcpsp_total_data.append(list(rcpsp_example.dependencies_deep_set))

    #     rcpsp_total_data.append(rcpsp_example.resources)

    #     # Write rcpsp data to its own file in the folder
    #     with open(os.path.join(output_folder, "rcpsp.json"), "w") as json_file:
    #         json.dump(rcpsp_total_data, json_file, indent=4)

    #     # Process Petri net data
    #     petri_total_data = []
    #     petri_total_data.append(len(petri_example.places))
    #     for place in petri_example.places:
    #         petri_total_data.append(place.__dict__)

    #     petri_total_data.append(len(petri_example.places_dict.values()))
    #     for val in petri_example.places_dict.values():
    #         petri_total_data.append(val.__dict__)

    #     petri_total_data.append(len(petri_example.transitions))
    #     for transition in petri_example.transitions:
    #         petri_total_data.append(transition.__dict__)

    #     petri_total_data.append(len(petri_example.transitions_dict.values()))
    #     for val in petri_example.transitions_dict.values():
    #         petri_total_data.append(val.__dict__)

    #     # Write petri data to its own file in the folder
    #     with open(os.path.join(output_folder, "petri.json"), "w") as json_file:
    #         json.dump(petri_total_data, json_file, indent=4)

    #     print(f"Successfully processed {folder_name}")
    #     print(f"Output saved to: {output_folder}")

    # except Exception as e:
    #     print(f"Error processing {file_path}: {str(e)}")

# def to_json():
#     base_path = "extract_problems/data/j30.sm.tgz"
#     output_base_dir = "json_outputs"

#     # Make sure the base output directory exists
#     os.makedirs(output_base_dir, exist_ok=True)

#     for group in range(1, 49):  # 48 groups
#         for index in range(1, 11):  # 10 files per group
#             # Create folder name based on the sm file
#             folder_name = f"j30{group}_{index}"
#             output_folder = os.path.join(output_base_dir, folder_name)

#             # Create a specific folder for this file
#             os.makedirs(output_folder, exist_ok=True)

#             # Construct the file path
#             file_name = f"{folder_name}.sm"
#             sm_file_path = f"{base_path}/{file_name}"

#             # Process the file
#             petri_example, rcpsp_example = init_real_problem(sm_file_path, True)

#             # Process RCPSP data
#             rcpsp_total_data = [act.__dict__ for act in rcpsp_example.activities]
#             rcpsp_total_data.append(rcpsp_example.backward_dependencies)
#             rcpsp_total_data.append(rcpsp_example.dependencies)
#             rcpsp_total_data.append(rcpsp_example.resources)

#             # Write rcpsp data to its own file in the folder
#             with open(os.path.join(output_folder, "rcpsp.json"), "w") as json_file:
#                 json.dump(rcpsp_total_data, json_file, indent=4)

#             # Process Petri net data
#             petri_total_data = []
#             petri_total_data.append(len(petri_example.places))
#             for place in petri_example.places:
#                 petri_total_data.append(place.__dict__)

#             petri_total_data.append(len(petri_example.places_dict.values()))
#             for val in petri_example.places_dict.values():
#                 petri_total_data.append(val.__dict__)

#             petri_total_data.append(len(petri_example.transitions))
#             for trasition in petri_example.transitions:
#                 petri_total_data.append(trasition.__dict__)

#             petri_total_data.append(len(petri_example.transitions_dict.values()))
#             for val in petri_example.transitions_dict.values():
#                 petri_total_data.append(val.__dict__)

#             # Write petri data to its own file in the folder
#             with open(os.path.join(output_folder, "petri.json"), "w") as json_file:
#                 json.dump(petri_total_data, json_file, indent=4)

#             print(f"Processed {folder_name}")

if __name__=="__main__":
    to_json()