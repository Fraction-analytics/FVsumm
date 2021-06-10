import h5py
filename = "/home/vinsent/h5datasets/video_1.mp4.h5"
f = h5py.File(filename, "r")
print(type(f))

for key in f.keys():
    print(key) #Names of the groups in HDF5 file

    #Get the HDF5 group
    # group2 = f.get(key+'/' + 'features/')
    # print(dir(group2))
    # print(group2)~
    seq = f[key]['features'][...] 
    print(seq)
    print(type(seq))
    print(seq.shape)
    group = f[key]
# print(group)
#Checkout what keys are inside that group.
for key in group.keys():
    print(key)
f.close()