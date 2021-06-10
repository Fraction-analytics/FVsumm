import h5py
filename = "/home/vinsent/Downloads/Torrient_fraction/dataset/eccv16_dataset_summe_google_pool5.h5"

with h5py.File(filename, "r") as f:
    # List all groups
    print("Keys: %s" % f.keys())
    a_group_key = list(f.keys())[0]

    # Get the data
    data = list(f[a_group_key])
