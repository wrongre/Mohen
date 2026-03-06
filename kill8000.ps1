$c = Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue
if ($c) {
  $c | ForEach-Object { Write-Output "KILLING:$($_.OwningProcess)"; Stop-Process -Id $_.OwningProcess -Force }
} else {
  Write-Output "NO_PORT_8000"
}
