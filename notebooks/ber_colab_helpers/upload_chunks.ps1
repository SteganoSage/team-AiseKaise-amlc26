# Uploads D:\ber_work\upload\chunks\* to the Colab session, 4 at a time, retrying failures.
$cli = "C:\Users\Saksham Gupta\Downloads\manual-sop-pipeline\.venv-colab\Scripts\python.exe"
$files = Get-ChildItem "D:\ber_work\upload\chunks" -File
$pending = [System.Collections.Queue]::new(@($files | ForEach-Object { @{ File = $_; Try = 0 } }))
$running = @()
$start = Get-Date
while ($pending.Count -gt 0 -or $running.Count -gt 0) {
    while ($running.Count -lt 4 -and $pending.Count -gt 0) {
        $item = $pending.Dequeue(); $item.Try++
        $job = Start-Job -ScriptBlock {
            param($cli, $path, $name)
            & $cli -m colab_cli.cli --auth=oauth2 upload -s ber $path "/content/ber_work/chunks/$name" 2>&1 | Out-String
        } -ArgumentList $cli, $item.File.FullName, $item.File.Name
        $running += @{ Job = $job; Item = $item }
    }
    Start-Sleep -Seconds 2
    foreach ($r in @($running)) {
        if ($r.Job.State -ne "Running") {
            $out = (Receive-Job $r.Job | Out-String).Trim()
            Remove-Job $r.Job
            $running = @($running | Where-Object { $_.Job.Id -ne $r.Job.Id })
            if ($out -match "Uploaded") {
                "{0:N0}s OK   {1}" -f ((Get-Date) - $start).TotalSeconds, $r.Item.File.Name
            } elseif ($r.Item.Try -lt 4) {
                "{0:N0}s RETRY {1} ({2})" -f ((Get-Date) - $start).TotalSeconds, $r.Item.File.Name, $out
                $pending.Enqueue($r.Item)
            } else {
                "{0:N0}s FAILED {1} ({2})" -f ((Get-Date) - $start).TotalSeconds, $r.Item.File.Name, $out
            }
        }
    }
}
"DONE in {0:N0}s" -f ((Get-Date) - $start).TotalSeconds
