# Polls the VM every 3 minutes until the pipeline finishes.
# The CLI's runtime-proxy token expires after ~1 h, after which the CLI wrongly
# prunes the session. So every poll first re-registers `ber` with a fresh token
# (restore_ber.py) and makes sure a keep-alive process is running.
$env:PYTHONIOENCODING = "utf-8"; $env:PYTHONUTF8 = "1"
$cli = "C:\Users\Saksham Gupta\Downloads\manual-sop-pipeline\.venv-colab\Scripts\python.exe"
$dir = "C:\Users\Saksham Gupta\Desktop\AmazonMLChallenge\business_entity_resolution\src\colab"
$endpoint = (Get-Content D:\ber_work\ber_endpoint.txt -Raw).Trim()
for ($i = 1; $i -le 60; $i++) {
    $restore = & $cli "$dir\restore_ber.py" 2>&1 | Out-String
    $alive = Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match "keep-alive $endpoint ber" }
    if (-not $alive) {
        Start-Process -FilePath $cli -ArgumentList "-m", "colab_cli.cli", "--auth=oauth2", "keep-alive", $endpoint, "ber" -WindowStyle Hidden | Out-Null
    }
    $out = & $cli -m colab_cli.cli --auth=oauth2 exec -s ber -f "$dir\tail.py" 2>&1 | Out-String
    "===== poll $i $(Get-Date -Format HH:mm:ss) | $($restore.Trim())`n$out" | Set-Content -Encoding utf8 D:\ber_work\vm_status.log
    if ($out -match "model_predict finished" -and $out -notmatch "rc=[1-9]") {
        Add-Content -Encoding utf8 D:\ber_work\vm_status.log "PIPELINE FINISHED - validating and downloading"
        & "$dir\finish_and_fetch.ps1"
        break
    }
    if ($out -match "rc=[1-9]" -or $restore -match "VM is gone") { break }
    Start-Sleep -Seconds 180
}
Add-Content -Encoding utf8 D:\ber_work\vm_status.log "POLLER DONE"
