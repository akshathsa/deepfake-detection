import pandas as pd

df = pd.read_csv("/global/cfs/projectdirs/m3641/Akaash/deepfake-detection/data/dfdc_test_labels.csv")
print(len(df[df['label'] == 0]))

# Train: 1537/339
# Val: 198/37
# Test: 179/56