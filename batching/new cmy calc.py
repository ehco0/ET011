import csv
import math

S_total = 2094.26023 
dx = 45.000
dz = -2.527

input_file = "/Users/butterscotch/Downloads/lf_data.csv"
output_file = "output_with_cmy_shift.csv"

with open(input_file, mode="r", encoding="latin-1", newline='') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

rows = [{k.strip(): v for k, v in r.items()} for r in rows]

configs = []
start_row = 0
while start_row < len(rows):
    configs.append(list(range(start_row, min(start_row + 16, len(rows)))))
    start_row += 16

for indices in configs:
    max_idx = max(indices, key=lambda i: float(rows[i]['LD']))
    max_row = rows[max_idx]

    AR_1wing = float(max_row['Aspect'])
    taper = float(max_row['Taper'])

    AR_total = 2 * AR_1wing
    span = math.sqrt(AR_total * S_total)
    c_root = 2 * S_total / (span * (1 + taper))
    MAC = (2/3) * c_root * (1 + taper + taper**2) / (1 + taper)

    for idx in indices:
        r = rows[idx]

        CL = float(r['CLtot'])
        CD = float(r['CDtot'])
        CMY_orig = float(r['CMytot'])

        CMY_new = CMY_orig + CL * (dx / MAC) - CD * (dz / MAC)

        r['CMY_new'] = CMY_new

fieldnames = list(rows[0].keys())
with open(output_file, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print("Output saved as:", output_file)
