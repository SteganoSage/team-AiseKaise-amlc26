# Downloads the packed output parts from the VM (4 in parallel, with retries),
# then rejoins, gunzips and verifies them into D:\ber_work\output.
$cli = "C:\Users\Saksham Gupta\Downloads\manual-sop-pipeline\.venv-colab\Scripts\python.exe"
$py = "C:\Users\Saksham Gupta\miniconda3\envs\bdh\python.exe"
$local = "D:\ber_work\download"
New-Item -ItemType Directory -Force $local | Out-Null
& $cli -m colab_cli.cli --auth=oauth2 download -s ber /content/ber_work/download/manifest.json "$local\manifest.json"
$manifest = Get-Content "$local\manifest.json" -Raw | ConvertFrom-Json
$parts = foreach ($f in $manifest.PSObject.Properties) { $f.Value.parts }
$pending = [System.Collections.Queue]::new(@($parts | ForEach-Object { @{ Name = $_; Try = 0 } }))
$running = @()
while ($pending.Count -gt 0 -or $running.Count -gt 0) {
    while ($running.Count -lt 4 -and $pending.Count -gt 0) {
        $item = $pending.Dequeue(); $item.Try++
        $job = Start-Job -ScriptBlock {
            param($cli, $name, $local)
            & $cli -m colab_cli.cli --auth=oauth2 download -s ber "/content/ber_work/download/$name" "$local\$name" 2>&1 | Out-String
        } -ArgumentList $cli, $item.Name, $local
        $running += @{ Job = $job; Item = $item }
    }
    Start-Sleep -Seconds 2
    foreach ($r in @($running)) {
        if ($r.Job.State -ne "Running") {
            $out = (Receive-Job $r.Job | Out-String).Trim(); Remove-Job $r.Job
            $running = @($running | Where-Object { $_.Job.Id -ne $r.Job.Id })
            $ok = Test-Path "$local\$($r.Item.Name)"
            if ($ok) { "OK    $($r.Item.Name)" }
            elseif ($r.Item.Try -lt 4) { "RETRY $($r.Item.Name) ($out)"; $pending.Enqueue($r.Item) }
            else { "FAILED $($r.Item.Name) ($out)" }
        }
    }
}
& $py -c @"
import gzip, hashlib, json, pathlib
d = pathlib.Path(r'$local'); out = pathlib.Path(r'D:\ber_work\output'); out.mkdir(parents=True, exist_ok=True)
for name, info in json.loads((d / 'manifest.json').read_text()).items():
    raw = gzip.decompress(b''.join((d / p).read_bytes() for p in info['parts']))
    ok = len(raw) == info['size'] and hashlib.md5(raw).hexdigest() == info['md5']
    (out / name).write_bytes(raw)
    print(name, f'{len(raw)/1e6:.1f} MB', 'checksum OK' if ok else 'CHECKSUM MISMATCH')
"@
