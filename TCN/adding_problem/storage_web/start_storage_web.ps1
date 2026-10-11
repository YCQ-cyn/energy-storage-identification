param(
    [switch]$Check
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$OutputEncoding = [Console]::OutputEncoding
$Host.UI.RawUI.WindowTitle = "Storage Screening Service"

$webRoot = $PSScriptRoot
$projectRoot = [IO.Path]::GetFullPath((Join-Path $webRoot ".."))
$repositoryRoot = [IO.Path]::GetFullPath((Join-Path $projectRoot "..\.."))
$sourceFile = Join-Path $webRoot "src\StorageScreeningApplication.java"
$buildRoot = Join-Path $webRoot "build"
$classFile = Join-Path $buildRoot "StorageScreeningApplication.class"
$errorLog = Join-Path $webRoot "startup-error.log"
$port = 8765

function Resolve-Executable {
    param([string]$Candidate)

    if ([string]::IsNullOrWhiteSpace($Candidate)) {
        return $null
    }
    if (Test-Path -LiteralPath $Candidate -PathType Leaf) {
        return (Resolve-Path -LiteralPath $Candidate).Path
    }
    $command = Get-Command $Candidate -CommandType Application -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if ($null -ne $command) {
        return $command.Source
    }
    return $null
}

function Get-JavaMajorVersion {
    param([string]$JavacExecutable)

    $versionText = (& $JavacExecutable -version 2>&1 | Out-String).Trim()
    if ($versionText -notmatch "(?<major>\d+)(?:\.(?<minor>\d+))?") {
        throw "Unable to detect the javac version: $versionText"
    }
    $major = [int]$Matches.major
    if ($major -eq 1 -and $Matches.minor) {
        $major = [int]$Matches.minor
    }
    return $major
}

function Test-PythonRuntime {
    param([string]$PythonExecutable)

    if ([string]::IsNullOrWhiteSpace($PythonExecutable)) {
        return $false
    }
    & $PythonExecutable -c "import numpy, pandas, torch, openpyxl" 2>$null
    return $LASTEXITCODE -eq 0
}

function Open-ServicePage {
    param([string]$Address)

    Start-Process $Address
    Write-Host "[OK] Opened $Address" -ForegroundColor Green
}

try {
    if ($env:STORAGE_WEB_PORT) {
        $configuredPort = 0
        if (-not [int]::TryParse($env:STORAGE_WEB_PORT, [ref]$configuredPort) -or
                $configuredPort -lt 1 -or $configuredPort -gt 65535) {
            throw "STORAGE_WEB_PORT must be an integer between 1 and 65535."
        }
        $port = $configuredPort
    }

    $address = "http://127.0.0.1:$port"
    $healthAddress = "$address/api/health"
    try {
        $health = Invoke-RestMethod -Uri $healthAddress -TimeoutSec 2
        if ($health.status -eq "ok") {
            Write-Host "[INFO] The service is already running." -ForegroundColor Yellow
            Open-ServicePage $address
            exit 0
        }
    } catch {
        # Continue when the health endpoint is unavailable.
    }

    $listener = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if ($null -ne $listener) {
        throw "Port $port is used by process $($listener.OwningProcess). Set STORAGE_WEB_PORT to use another port."
    }

    foreach ($requiredPath in @(
        $sourceFile,
        (Join-Path $projectRoot "add_test_infer.py"),
        (Join-Path $projectRoot "train_student\best_model_student.pt"),
        (Join-Path $webRoot "static\index.html")
    )) {
        if (-not (Test-Path -LiteralPath $requiredPath)) {
            throw "Required file is missing: $requiredPath"
        }
    }

    $java = Resolve-Executable "java"
    $javac = Resolve-Executable "javac"
    if ($null -eq $java -or $null -eq $javac) {
        throw "JDK was not found. Install JDK 17 or newer and add java and javac to PATH."
    }
    $javaMajor = Get-JavaMajorVersion $javac
    if ($javaMajor -lt 17) {
        throw "JDK $javaMajor is too old. JDK 17 or newer is required."
    }

    $pythonCandidates = @(
        $env:STORAGE_PYTHON,
        "D:\Anaconda\python.exe",
        "python"
    )
    $python = $null
    foreach ($candidate in $pythonCandidates) {
        $resolved = Resolve-Executable $candidate
        if ($resolved -and (Test-PythonRuntime $resolved)) {
            $python = $resolved
            break
        }
    }
    if ($null -eq $python) {
        throw "No Python runtime contains numpy, pandas, torch and openpyxl. Set STORAGE_PYTHON to python.exe."
    }

    $needsCompile = -not (Test-Path -LiteralPath $classFile) -or
        (Get-Item -LiteralPath $sourceFile).LastWriteTimeUtc -gt
        (Get-Item -LiteralPath $classFile).LastWriteTimeUtc
    if ($needsCompile) {
        Write-Host "[1/2] Compiling the Java service..."
        New-Item -ItemType Directory -Path $buildRoot -Force | Out-Null
        & $javac --add-modules jdk.httpserver -encoding UTF-8 -d $buildRoot $sourceFile
        if ($LASTEXITCODE -ne 0) {
            throw "Java compilation failed with exit code $LASTEXITCODE."
        }
    } else {
        Write-Host "[1/2] Java compilation is up to date."
    }

    if ($Check) {
        Write-Host "[OK] Runtime, files, port and Java compilation are ready." -ForegroundColor Green
        exit 0
    }

    if (Test-Path -LiteralPath $errorLog) {
        Remove-Item -LiteralPath $errorLog -Force
    }
    Write-Host "[2/2] Starting the storage screening service..."
    Write-Host "[JDK] $javaMajor"
    Write-Host "[Python] $python"
    Write-Host "[URL] $address"
    Write-Host "Close this window to stop the service." -ForegroundColor Cyan

    if ($env:PYTHONPATH) {
        $env:PYTHONPATH = "$repositoryRoot;$env:PYTHONPATH"
    } else {
        $env:PYTHONPATH = $repositoryRoot
    }

    & $java --add-modules jdk.httpserver "-Dfile.encoding=UTF-8" -cp $buildRoot `
        StorageScreeningApplication --root $projectRoot --python $python --port $port
    if ($LASTEXITCODE -ne 0) {
        throw "The Java service exited with code $LASTEXITCODE."
    }
} catch {
    $message = $_.Exception.Message
    $details = "{0}`r`n{1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), ($_ | Out-String)
    [IO.File]::WriteAllText($errorLog, $details, [Text.UTF8Encoding]::new($false))
    Write-Host "[ERROR] $message" -ForegroundColor Red
    Write-Host "Details: $errorLog" -ForegroundColor DarkGray
    exit 1
}
