"""Pipeline progress on the VM: log tail, memory, running stage."""
import subprocess

print(subprocess.run(
    "tail -n 25 /content/ber_work/full_run.log; echo; free -g | head -2; "
    "echo; ps -eo pid,etime,pcpu,rss,args --sort=-rss | grep -E 'ber\\.|run_all' | grep -v grep | head -5; "
    "echo; du -sh /content/ber_work/* 2>/dev/null",
    shell=True, capture_output=True, text=True).stdout)
