import os

for d in ("/content/ber_work/chunks", "/content/ber_work/raw"):
    os.makedirs(d, exist_ok=True)
print(sorted(os.listdir("/content/ber_work")), sorted(os.listdir("/content/ber_work/raw")))
