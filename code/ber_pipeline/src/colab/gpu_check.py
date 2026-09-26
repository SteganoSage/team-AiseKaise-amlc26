import subprocess
import time

import torch
import transformers

print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
print("transformers", transformers.__version__)
from transformers import AutoModelForSequenceClassification, AutoTokenizer

t = time.time()
name = "intfloat/multilingual-e5-small"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=1).cuda().to(torch.bfloat16)
print("loaded in", round(time.time() - t, 1), "s; params", sum(p.numel() for p in model.parameters()) // 1_000_000, "M")
a = ["Shakti Builders Private Limited | Flat No A-901, Hyderabad, Telangana"] * 4096
b = ["శక్తి బిల్డర్స్ ప్రైవేట్ లిమిటెడ్ | FLAT NO A-5-901, RANGAREDDY, HYDERABAD"] * 4096
enc = tok(a, b, truncation=True, max_length=96, padding=True, return_tensors="pt").to("cuda")
with torch.no_grad():
    model(**enc)
    torch.cuda.synchronize(); t = time.time()
    for _ in range(5):
        model(**{k: v for k, v in enc.items()})
    torch.cuda.synchronize()
print("inference pairs/s:", round(5 * 4096 / (time.time() - t)), "seq len", enc["input_ids"].shape[1])
print(subprocess.run("pip list 2>/dev/null | grep -iE '^(torch|transformers|accelerate|tokenizers) '",
                     shell=True, capture_output=True, text=True).stdout)
