import matplotlib.pyplot as plt
import numpy as np
# import matplotlib.dates as mdates
from datetime import datetime
dates = ['1688087730.719562', '1688087230.719562', '1688087770.719562', '1688087930.719562',]
dates = [datetime.fromtimestamp(float(VAR)).strftime("%I:%M:%S") for VAR in dates]
names = ['v2.2.4', 'v3.0.3', 'v3.0.2', 'v3.0.1',]

fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
ax.plot(dates, np.zeros_like(dates), "-o",
        color="k", markerfacecolor="w")  # Baseline and markers on it.

plt.show()