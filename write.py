import hdf5plugin
import h5py
import numpy as np


def string_dt(s):
    tid = h5py.h5t.C_S1.copy()
    tid.set_size(len(s))
    return h5py.Datatype(tid)

def create_string_attr(obj, key, value):
    # Albula requires null terminated strings
    if value[-1] != "\x00":
        value += "\x00"
    obj.attrs.create(key, value, dtype=string_dt(value))



filename = "test.h5"
mode = "w"
image_size = (512,1024)
dtype = np.float32

#Current frame and also number of frames in dataset
frame_index = 0 


#Create a new file and dataset
f = h5py.File(filename, mode)
compression = hdf5plugin.Bitshuffle(nelems=0, cname='lz4')
nxentry = f.create_group("entry")
create_string_attr(nxentry, "NX_class", "NXentry")
nxdata = nxentry.create_group("data")
create_string_attr(nxdata, "NX_class", "NXdata")
ds = nxdata.create_dataset(
    "data_000001",
    shape=(0, *image_size),
    dtype=dtype,
    maxshape=(None, *image_size),
    chunks=(1, *image_size),
    **compression,
)

#Extend the dataset and write some data
image = np.zeros(image_size, dtype=dtype)
ds.resize(frame_index + 1, axis=0)
ds[frame_index] = image
frame_index += 1

print(ds.shape)

f.close()