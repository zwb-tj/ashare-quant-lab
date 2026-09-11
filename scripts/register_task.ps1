# 注册 Windows 计划任务：每个交易日 17:30 运行 aqlab 每日选股并推送飞书
# 用法（普通用户权限即可）：powershell -ExecutionPolicy Bypass -File scripts\register_task.ps1
# 卸载：schtasks /Delete /TN "aqlab-daily" /F

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$runner = Join-Path $ProjectRoot 'scripts\run_daily.ps1'
$taskName = 'aqlab-daily'

if (-not (Test-Path $runner)) { throw "找不到 $runner" }

$action = New-ScheduledTaskAction -Execute 'powershell.exe' `
    -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$runner`"" `
    -WorkingDirectory $ProjectRoot

# 周一到周五 17:30（节假日也会触发，数据源无新数据时报告会提示）
$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At '17:30'

$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Minutes 30)

Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Settings $settings `
    -Description 'aqlab 每日选股：tushare 拉数 → 规则打分 → 飞书推送' -Force | Out-Null

Write-Host "已注册计划任务：$taskName（每周一至周五 17:30）"
Write-Host "手动试跑：powershell -ExecutionPolicy Bypass -File `"$runner`" -DryRun"
Write-Host "查看任务：schtasks /Query /TN $taskName"
