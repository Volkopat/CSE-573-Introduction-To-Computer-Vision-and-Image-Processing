import json

json_list = [] #each element is a dictionary, {"iname": "1.jpg", "bbox": [1, 2, 3 ,5]}

element_1 = {"iname": "1.jpg", "bbox": [1, 2, 3, 5]} #first element in json file
element_2 = {"iname": "1.jpg", "bbox": [10, 20, 30, 40]} #second element in json file
element_3 = {"iname": "2.jpg", "bbox": [100, 120, 35, 45]} #third element in json file

#add element to list
json_list.append(element_1)
json_list.append(element_2)
json_list.append(element_3)

#the result json file name
output_json = "results1.json"
#dump json_list to result.json
with open(output_json, 'w') as f:
    json.dump(json_list, f)


