import pandas as pd
import xarray as xr
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
os.chdir(r'C:\Users\rampaln\OneDrive - NIWA\Repositories\rip_forecasting\rip-detection')
data = xr.open_dataset(r'pauanui_record.nc')

subset = data.sel(time =(data.time.to_index().hour > 8) &
                        (data.time.to_index().hour < 16))
dset = subset.isel(x = slice(50, 100), y =slice(50, 100))['rip_images']
mask = dset.min(["x","y","channel"])
fig, ax = plt.subplots()
ax.plot(dset.time.to_index()[mask], dset.isel(channel =0).values[mask], 'rx')
fig.show()
# Lots of bad data need to screen through the data well
# Potentiall higher resolution required
# data is actually too coarse at this resolution
# Need to make data at a higer resolution to actually resolve these rips

# Very oblique angles could be unideal - need actually a better test case
# need better thresholding and testing, optimal range of values seems be 5

# Could train a simple anomaly detection algorithm using this method and deep learning?
# This might help us actually get through and mask out parts of images that are ppor


mask1 = np.where((mask.values >18.0) &(mask.values <95.0))[0]
# images in the above range seem to be algood
data = subset.isel(time = mask1)
model = tf.keras.models.load_model('best_model.h5')
preds = model.predict(data['rip_images'],verbose =1, batch_size =82)
# should take about 15 seconds to classify images
df = pd.DataFrame(data = np.argmax(preds, axis =-1),
                  index = data.time.to_index(), columns=['predictions'])