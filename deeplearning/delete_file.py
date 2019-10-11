import os
base_path = '/home/zlw/dataset/SAMM_FACE_CUT/SAMM/006/7/'
start = 9438
end = 9578
for i in range(141):
    temp_path = base_path + '006_' + str(start+i) + '.jpg'
    print(temp_path)
    os.remove(temp_path)
print('done')