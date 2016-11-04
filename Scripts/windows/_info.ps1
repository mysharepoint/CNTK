﻿#
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
#

function FunctionIntro(
    [Parameter(Mandatory = $true)][hashtable] $table
)
{
    $table | Out-String | Write-Verbose
}

function GetKey(
    [string] $validChar
)
{
    do {
        $key = Read-Host
    } until ($key -match $validChar)

    return $key
}

function DisplayStartMessage
{
"

This script will setup the CNTK prequisites and the CNTK Python environment onto the machine.
More help is given by calling 'get-help .\install.ps1' in your powershell environment.

The script will analyse your machine and will determine which components are required. 
The required components will be downloaded in [$localCache]
Repeated operation of this script will reuse already downloaded components.

 - If required VS2012 Runtime and VS2013 Runtime will be installed
 - If required MSMPI will be installed
 - If required the standard Git tool will be installed
 - CNTK source will be cloned from Git into [$RepoLocation]
 - Anaconda3 will be installed into [$AnacondaBasePath]
 - A CNTK-PY34 environment will be created in [$AnacondaBasePath\envs]
 - CNTK will be installed into the CNTK-PY34 environment
"
}

function DisplayVersionWarningMessage(
    [string] $version)
{
"
You are executing this script from Powershell Version $version.
We recommend that you execute the script from Powershell Version 4 or later. You can install Powershell Version 4 from:
    https://www.microsoft.com/en-us/download/details.aspx?id=40855
"
}

function DisplayWarningNoExecuteMessage
{
"
The parameter '-Execute' hasn't be supplied to the script.
The script will execute withouth making any actual changes to the machine.
"
}

function DisplayStartContinueMessage
{
"
1 - I agree and want to continue
Q - Quit the installation process
"
}

function CheckPowershellVersion
{
    $psVersion = $PSVersionTable.PSVersion.Major
    if ($psVersion -ge 4) {
        return $true
    }

    Write-Host $(DisplayVersionWarningMessage $psVersion)
    if ($psVersion -eq 3) {
        return $true
    }
    return $false
}

function CheckOSVersion 
{
    $runningOn = (Get-WmiObject -class Win32_OperatingSystem).Caption
    $isMatching = ($runningOn -match "^Microsoft Windows (8\.1|10|Server 2012 R2)") 

    if ($isMatching) {
        return
    }

    Write-Host "
You are running the this install script on [$runningOn].
The Microsoft Cognitive Toolkit is designed and tested on Windows 8.1, Windows 10, 
and Windows Server 2012 R2. You can continue with the installation, thought some 
features of CNTK may not be working as expected.
"
    return
}

function DisplayStart()
{
    Write-Host $(DisplayStartMessage)

    if (-not (CheckPowershellVersion)) {
        return $false
    }

    CheckOSVersion

    if (-not $Execute) {
        Write-Host $(DisplayWarningNoExecuteMessage)
    }
    
    Write-Host $(DisplayStartContinueMessage)
    $choice = GetKey '^[1qQ]+$'

    if ($choice -contains "1") {
        return $true
    }

    return $false
}


Function DisplayEnd() 
{
if (-not $Execute) { return }

Write-Host "

CNTK v2 Python install complete.

To activate the CNTK Python environment and set the PATH to include CNTK, start a command shell and run
   $cntkRootDir\scripts\cntkpy34.bat


Please checkout examples in the CNTK repository clone here:
    c:\repos\cntk\bindings\python\examples

"
}
