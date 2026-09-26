"""On the VM: replace /content/ber with the freshly uploaded /content/code.zip."""
import shutil
import zipfile

shutil.rmtree("/content/ber/src", ignore_errors=True)
zipfile.ZipFile("/content/code.zip").extractall("/content/ber")
print(sorted(p for p in zipfile.ZipFile("/content/code.zip").namelist() if p.startswith("src/ber/")))
