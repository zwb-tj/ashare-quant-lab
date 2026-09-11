# aqlab 每日选股任务（Windows PowerShell）
# 用法：
#   powershell -ExecutionPolicy Bypass -File scripts\run_daily.ps1            # 真实推送
#   powershell -ExecutionPolicy Bypass -File scripts\run_daily.ps1 -DryRun    # 只打印不推送
# 依赖：项目已 `pip install -e .`，且 .env 中配置了 TUSHARE_TOKEN 与 FEISHU_WEBHOOK

param(
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $ProjectRoot

# ---- 读取 .env（简单 KEY=VALUE 解析，忽略注释与空行）----
$envFile = Join-Path $ProjectRoot '.env'
if (Test-Path $envFile) {
    Get-Content $envFile | ForEach-Object {
        $line = $_.Trim()
        if ($line -and -not $line.StartsWith('#') -and $line.Contains('=')) {
            $name, $value = $line.Split('=', 2)
            [System.Environment]::SetEnvironmentVariable($name.Trim(), $value.Trim(), 'Process')
        }
    }
} else {
    Write-Warning "未找到 .env（可先复制 .env.example），将使用合成数据 dry-run。"
}

# ---- 日志 ----
$logDir = Join-Path $ProjectRoot 'output\logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$stamp = Get-Date -Format 'yyyy-MM-dd'
$logFile = Join-Path $logDir "daily-$stamp.log"

# ---- 组装命令 ----
$argsList = @('-m', 'aqlab.cli', 'daily', '--top', '10', '--out', 'output')
if ($env:AQLAB_SYMBOLS -and $env:TUSHARE_TOKEN) {
    $startDate = if ($env:AQLAB_START) { $env:AQLAB_START } else { '2023-01-01' }
    $argsList += @('--tushare', '--symbols', $env:AQLAB_SYMBOLS, '--start', $startDate)
} else {
    Write-Warning '未配置 TUSHARE_TOKEN/AQLAB_SYMBOLS，改用合成数据（仅用于验证流程）。'
}
if ($env:AQLAB_RULE_OVERRIDES) {
    foreach ($item in $env:AQLAB_RULE_OVERRIDES.Split(',')) {
        if ($item.Trim()) { $argsList += @('--rule', $item.Trim()) }
    }
}
if (-not $DryRun) { $argsList += '--push' }

# ---- 运行并落日志（兼容 Windows PowerShell 5.1）----
$stampLine = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] aqlab daily 开始：python $($argsList -join ' ')"
$stampLine | Add-Content -Path $logFile -Encoding UTF8
$output = & python @argsList 2>&1
$output | Out-String | Add-Content -Path $logFile -Encoding UTF8
$output | Write-Host
$doneLine = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] 结束（exit=$LASTEXITCODE）"
$doneLine | Add-Content -Path $logFile -Encoding UTF8
exit $LASTEXITCODE
