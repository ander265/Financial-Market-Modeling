import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import CAPMautoscrape

dfr = dfr[dfr['Date'].isin(dfs[0].iloc[:,0])].reset_index(drop=True)