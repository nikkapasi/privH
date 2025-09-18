import numpy as np

rows = 32
cols = 48

host_in = np.zeros((rows, cols), dtype=np.float32)

for r in range(rows):
    for c in range(cols):
        host_in[r, c] = 0.01 * r + 0.1 * c + 1.0

np.savetxt("hamming/test_data.txt", host_in.flatten(), fmt='%.6f')

print(f"Generated test data with {rows} rows and {cols} columns, saved to hamming/test_data.txt")
