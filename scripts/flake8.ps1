# PowerShell wrapper to run flake8 with explicit Python version
param(
    [Parameter(ValueFromRemainingArguments=$true)]
    $args
)
py -3.14 -m flake8 @args
