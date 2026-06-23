# comfortably read h5 file
import h5py
h5_path="/home/csl/genie-sim/genie_sim-origin/source/data_collection/recording_data/[sort_the_fruit_into_the_box_apple_g2_7]/aligned_joints_all.h5"

def print_h5_structure(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"Dataset: {name}, shape: {obj.shape}, dtype: {obj.dtype}")
        elif isinstance(obj, h5py.Group):
            print(f"Group: {name}")

try:
    with h5py.File(h5_path, 'r') as f:
        print("HDF5 file structure:")
        f.visititems(print_h5_structure)

except FileNotFoundError:
    print(f"File not found: {h5_path}")

except Exception as e:
    print(f"Error reading HDF5 file: {e}")