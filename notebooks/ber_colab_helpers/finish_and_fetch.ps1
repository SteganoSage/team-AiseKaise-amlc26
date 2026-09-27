# After the pipeline finishes on the VM: run the official validator and pack the
# outputs there, then download them to D:\ber_work\output. Called by poll.ps1 so
# results leave the VM as soon as they exist.
$env:PYTHONIOENCODING = "utf-8"; $env:PYTHONUTF8 = "1"
$cli = "C:\Users\Saksham Gupta\Downloads\manual-sop-pipeline\.venv-colab\Scripts\python.exe"
$dir = "C:\Users\Saksham Gupta\Desktop\AmazonMLChallenge\business_entity_resolution\src\colab"
$log = "D:\ber_work\finish_fetch.log"
"started $(Get-Date -Format HH:mm:ss)" | Set-Content -Encoding utf8 $log
& $cli "$dir\restore_ber.py" *>> $log
"import os; os.makedirs('/content/ber_scripts', exist_ok=True); os.makedirs('/content/utils', exist_ok=True); print('ok')" |
    Set-Content -Encoding ascii D:\ber_work\upload\mk_dirs2.py
& $cli -m colab_cli.cli --auth=oauth2 exec -s ber -f D:\ber_work\upload\mk_dirs2.py *>> $log
& $cli -m colab_cli.cli --auth=oauth2 upload -s ber "C:\Users\Saksham Gupta\Desktop\AmazonMLChallenge\student_resource\utils\validate_submission.py" /content/utils/validate_submission.py *>> $log
foreach ($f in "validate_vm.py", "pack_outputs.py") {
    & $cli -m colab_cli.cli --auth=oauth2 upload -s ber "$dir\$f" "/content/ber_scripts/$f" *>> $log
}
& $cli -m colab_cli.cli --auth=oauth2 exec -s ber -f "$dir\finish.py" *>> $log
for ($i = 1; $i -le 40; $i++) {
    Start-Sleep -Seconds 20
    & $cli "$dir\restore_ber.py" | Out-Null
    $out = & $cli -m colab_cli.cli --auth=oauth2 exec -s ber -f "$dir\tail_finish.py" 2>&1 | Out-String
    if ($out -match "FINISH_EXIT") { $out | Add-Content -Encoding utf8 $log; break }
}
& $cli "$dir\restore_ber.py" | Out-Null
& "$dir\fetch_outputs.ps1" *>> $log
"done $(Get-Date -Format HH:mm:ss)" | Add-Content -Encoding utf8 $log
