import os
files=os.listdir('./')
files.remove('json2dataset.py')
for i in range(len(files)):
    os.system('labelme_json_to_dataset ' + f'-o ../labelme_json/{files[i][:-5]}_json ' + files[i])
    print('Finish: ', files[i])
