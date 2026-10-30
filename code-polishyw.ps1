$CodePolishyManagedWrapper = $true
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Fail([string]$Message) { [Console]::Error.WriteLine("code-polishyw: $Message"); exit 1 }
function Show-Usage { Write-Output 'Usage: .\code-polishyw.ps1 setup [--source PATH]'; Write-Output '       .\code-polishyw.ps1 COMMAND [ARG...]' }
function Save-Archive([Uri]$Uri, [string]$Path, [long]$ExpectedSize) {
  Add-Type -AssemblyName System.Net.Http
  $Handler = [System.Net.Http.HttpClientHandler]::new()
  $Handler.AllowAutoRedirect = $false
  $Client = [System.Net.Http.HttpClient]::new($Handler)
  $Client.Timeout = [TimeSpan]::FromMinutes(10)
  $Current = $Uri
  $Response = $null
  try {
    for ($Redirects = 0; $Redirects -le 10; $Redirects++) {
      $Response = $Client.GetAsync($Current, [System.Net.Http.HttpCompletionOption]::ResponseHeadersRead).GetAwaiter().GetResult()
      $Status = [int]$Response.StatusCode
      if ($Status -lt 300 -or $Status -gt 399) { break }
      if ($Redirects -eq 10 -or $null -eq $Response.Headers.Location) { Fail 'release archive download returned an invalid redirect' }
      $Current = [Uri]::new($Current, $Response.Headers.Location)
      if ($Current.Scheme -cne 'https' -or $Current.UserInfo -or $Current.Fragment) { Fail 'release archive download redirected outside HTTPS' }
      $Response.Dispose()
      $Response = $null
    }
    if ($null -eq $Response -or [int]$Response.StatusCode -ne 200) { Fail 'release archive download failed' }
    $Length = $Response.Content.Headers.ContentLength
    if ($null -ne $Length -and [long]$Length -ne $ExpectedSize) { Fail 'release archive size mismatch' }
    $InputStream = $Response.Content.ReadAsStreamAsync().GetAwaiter().GetResult()
    $OutputStream = [System.IO.File]::Open($Path, [System.IO.FileMode]::CreateNew, [System.IO.FileAccess]::Write, [System.IO.FileShare]::None)
    [long]$Written = 0
    try {
      $Buffer = New-Object byte[] 1048576
      while (($Read = $InputStream.Read($Buffer, 0, $Buffer.Length)) -gt 0) {
        $Written += $Read
        if ($Written -gt $ExpectedSize) { Fail 'release archive size mismatch' }
        $OutputStream.Write($Buffer, 0, $Read)
      }
    } finally {
      $OutputStream.Dispose()
      $InputStream.Dispose()
    }
  } finally {
    if ($null -ne $Response) { $Response.Dispose() }
    $Client.Dispose()
    $Handler.Dispose()
  }
  if ($Written -ne $ExpectedSize) { Fail 'release archive size mismatch' }
}

$Arguments = @($args)
$Repository = [System.IO.Path]::GetFullPath($PSScriptRoot)
$LockPath = Join-Path $Repository '.code-polishy.lock.json'
if (-not (Test-Path -LiteralPath $LockPath -PathType Leaf)) { Fail '.code-polishy.lock.json must be a regular file' }
$LockItem = Get-Item -LiteralPath $LockPath -Force
if (($LockItem.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0 -or $LockItem.Length -le 0 -or $LockItem.Length -gt 32768) { Fail '.code-polishy.lock.json is unsafe or exceeds the wrapper input limit' }
$LockText = Get-Content -LiteralPath $LockPath -Raw
foreach ($Key in @('lockVersion', 'codePolishyVersion', 'releaseDigest')) {
  if ([regex]::Matches($LockText, '"' + [regex]::Escape($Key) + '"\s*:').Count -ne 1) { Fail ".code-polishy.lock.json must contain exactly one $Key field" }
}
try { $Lock = $LockText | ConvertFrom-Json } catch { Fail '.code-polishy.lock.json is not valid JSON' }
if ((($Lock.lockVersion -isnot [int]) -and ($Lock.lockVersion -isnot [long])) -or $Lock.codePolishyVersion -isnot [string] -or $Lock.releaseDigest -isnot [string]) { Fail '.code-polishy.lock.json has invalid release identity types' }
$LockVersion = [int]$Lock.lockVersion
$Version = [string]$Lock.codePolishyVersion
$Digest = [string]$Lock.releaseDigest
if ($LockVersion -ne 2) { Fail '.code-polishy.lock.json uses an unsupported lockVersion' }
if ($Version -cnotmatch '^[0-9A-Za-z][0-9A-Za-z._+-]*$' -or $Digest -cnotmatch '^[0-9a-f]{64}$') { Fail '.code-polishy.lock.json has an invalid release identity' }
if (-not $env:LOCALAPPDATA) { Fail 'LOCALAPPDATA is required to locate the shared Code Polishy installation' }
$Prefix = Join-Path $env:LOCALAPPDATA 'CodePolishy'
$ReleaseRoot = Join-Path (Join-Path $Prefix 'releases') "$Version-$Digest"
$Launcher = Join-Path (Join-Path $ReleaseRoot 'bin') 'code-polishy-launcher.exe'

function Test-Available {
  if (-not (Test-Path -LiteralPath $ReleaseRoot -PathType Container) -or -not (Test-Path -LiteralPath $Launcher -PathType Leaf)) { return $false }
  if (((Get-Item -LiteralPath $ReleaseRoot -Force).Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0 -or ((Get-Item -LiteralPath $Launcher -Force).Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) { return $false }
  return $true
}

function Test-Ready {
  if (-not (Test-Available)) { return $false }
  & $Launcher --repo-root $Repository version *> $null
  return $LASTEXITCODE -eq 0
}

if ($Arguments.Count -eq 0) { [Console]::Error.WriteLine((Show-Usage | Out-String).TrimEnd()); exit 2 }
if ($Arguments[0] -cne 'setup') {
  if (-not (Test-Available)) { Fail 'the locked release is not installed; run .\code-polishyw.ps1 setup' }
  & $Launcher --repo-root $Repository @Arguments
  exit $LASTEXITCODE
}

$Source = ''
$Index = 1
while ($Index -lt $Arguments.Count) {
  $Argument = $Arguments[$Index]
  if ($Argument -ceq '--source') {
    if ($Index + 1 -ge $Arguments.Count) { Fail '--source requires a local checkout' }
    $Source = $Arguments[$Index + 1]
    $Index += 2
  } elseif ($Argument.StartsWith('--source=', [System.StringComparison]::Ordinal)) {
    $Source = $Argument.Substring(9)
    $Index++
  } elseif ($Argument -in @('--help', '-h')) {
    Show-Usage
    exit 0
  } else { Fail "unknown setup option: $Argument" }
}
if (Test-Ready) { Write-Output "Code Polishy $Version is already ready for this repository."; exit 0 }

if ($Source) {
  try { $Source = (Resolve-Path -LiteralPath $Source).Path } catch { Fail '--source must name a local checkout' }
  $SourceVersionPath = Join-Path $Source 'VERSION'
  if (-not (Test-Path -LiteralPath $SourceVersionPath -PathType Leaf) -or (Get-Content -LiteralPath $SourceVersionPath -Raw).TrimEnd([char[]]"`r`n") -cne $Version) { Fail 'the source checkout VERSION does not match the repository lock' }
  $ToolInstaller = Join-Path $Source 'tools\install-policy-tools.ps1'
  $ReleaseInstaller = Join-Path $Source 'scripts\install.ps1'
  if (-not (Test-Path -LiteralPath $ToolInstaller -PathType Leaf) -or -not (Test-Path -LiteralPath $ReleaseInstaller -PathType Leaf)) { Fail 'the source checkout has no installers' }
  & $ToolInstaller
  if ($LASTEXITCODE -ne 0) { Fail 'policy-tool installation failed' }
  & $ReleaseInstaller -Prefix $Prefix -RequireRepository $Repository
  if ($LASTEXITCODE -ne 0) { Fail 'Code Polishy installation failed' }
} else {
  if ($env:PROCESSOR_ARCHITECTURE -ne 'AMD64') { Fail 'there is no Code Polishy release for this host' }
  $Archives = @($Lock.publication.archives | Where-Object { $_.host -ceq 'windows-x64' })
  if ($Archives.Count -ne 1) { Fail 'the lock does not contain one archive for this host' }
  if ($Archives[0].url -isnot [string] -or $Archives[0].sha256 -isnot [string] -or (($Archives[0].size -isnot [int]) -and ($Archives[0].size -isnot [long]))) { Fail 'the lock contains invalid archive metadata types' }
  $ArchiveURL = [string]$Archives[0].url
  $ArchiveSHA = [string]$Archives[0].sha256
  $ArchiveSize = [long]$Archives[0].size
  $ParsedURL = $null
  if (-not [Uri]::TryCreate($ArchiveURL, [UriKind]::Absolute, [ref]$ParsedURL) -or $ParsedURL.Scheme -cne 'https' -or $ParsedURL.UserInfo -or $ParsedURL.Fragment -or $ArchiveSHA -cnotmatch '^[0-9a-f]{64}$' -or $ArchiveSize -le 0 -or $ArchiveSize -gt 4294967296) { Fail 'the lock contains invalid archive metadata' }
  $Scratch = Join-Path ([System.IO.Path]::GetTempPath()) ("code-polishyw-" + [guid]::NewGuid().ToString('N'))
  New-Item -ItemType Directory -Path $Scratch | Out-Null
  try {
    $Archive = Join-Path $Scratch 'release.zip'
    Save-Archive $ParsedURL $Archive $ArchiveSize
    if ((Get-FileHash -Algorithm SHA256 $Archive).Hash.ToLowerInvariant() -cne $ArchiveSHA) { Fail 'release archive checksum mismatch' }
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $BootstrapRoot = Join-Path $Scratch 'release'
    $BootstrapDirectory = Join-Path $BootstrapRoot 'bin'
    New-Item -ItemType Directory -Path $BootstrapDirectory | Out-Null
    $Bootstrap = Join-Path $BootstrapDirectory 'code-polishy.exe'
    $Zip = [System.IO.Compression.ZipFile]::OpenRead($Archive)
    try {
      $Entries = @($Zip.Entries | Where-Object { $_.FullName -ceq 'bin/code-polishy.exe' })
      if ($Entries.Count -ne 1 -or $Entries[0].Length -le 0 -or $Entries[0].Length -gt 268435456) { Fail 'release archive has an invalid installer bootstrap' }
      [System.IO.Compression.ZipFileExtensions]::ExtractToFile($Entries[0], $Bootstrap)
    } finally {
      $Zip.Dispose()
    }
    & $Bootstrap --policy-root $BootstrapRoot install-bundle --source $Archive --sha256 $ArchiveSHA --prefix $Prefix
    if ($LASTEXITCODE -ne 0) { Fail 'Code Polishy installation failed' }
  } finally {
    if (Test-Path -LiteralPath $Scratch) { Remove-Item -LiteralPath $Scratch -Recurse -Force }
  }
}

if (-not (Test-Ready)) { Fail 'installation completed without making the exact locked release available' }
Write-Output "Code Polishy $Version is ready for this repository."
