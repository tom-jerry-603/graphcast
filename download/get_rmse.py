import h5py
import xarray as xr
import numpy as np

# === Load HDF5 file ===
with h5py.File("result.h5", "r") as f:
    # Example: f["fields"][time_idx, channel_idx, lat, lon]
    h5_data = f["forecast"][:]  # Shape: (T, C, H, W)
    h5_var = h5_data[::6, 0, :, :]  # Assuming channel 0 is the variable of interest
    # h5_var = np.concatenate([h5_var[..., 720:], h5_var[..., :720]], axis=-1)


# === Load NetCDF file ===
ds = xr.open_dataset("download/real-temp.nc")
nc_var = ds["t2m"].values  # Replace with actual variable name
nc_var = nc_var[::6, ::-1, :]
nc_var = nc_var[:, :-1, :]
print(nc_var.shape)

print(h5_var[:20, 120, 0])
print(nc_var[:20, 120, 0])
# === Align shapes (optional: depends on input) ===

if h5_var.shape != nc_var.shape:
    print("Shape mismatch:", h5_var.shape, "vs", nc_var.shape)
    # Resample, interpolate, or slice as needed here

# === Compute RMSE ===
rmse = np.sqrt(np.mean((h5_var[:20, :, :] - nc_var[:20, :, :]) ** 2))
print("RMSE:", rmse)