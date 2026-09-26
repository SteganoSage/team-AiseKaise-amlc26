import subprocess

print(subprocess.run("cat /content/ber_work/finish.log", shell=True, capture_output=True,
                     text=True).stdout[-3000:])
