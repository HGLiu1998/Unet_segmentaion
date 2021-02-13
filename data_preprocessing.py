import os

file_path = './json_file/'
json_path = './config/'
json_files = os.listdir(file_path)
for json in json_files:
    print(''.join(json.split('.')[:-1]))
    os.system("labelme_json_to_dataset {} -o {}".format(file_path + json, json_path + ''.join(json.split('.')[:-1])))