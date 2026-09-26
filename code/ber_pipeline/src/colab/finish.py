"""On the VM: validate + pack outputs as a background process (log: finish.log)."""
import subprocess

cmd = ("python /content/ber_scripts/validate_vm.py && python /content/ber_scripts/pack_outputs.py; "
       "echo FINISH_EXIT $?")
log = open("/content/ber_work/finish.log", "w")
p = subprocess.Popen(cmd, shell=True, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
print("finish started", p.pid)
