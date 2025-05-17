# PowerShell script to fix Python files for Linux compatibility
# Converts CRLF to LF and removes null bytes

Write-Host "Fixing Python files for Linux compatibility..." -ForegroundColor Green

$count = 0
$fixedFiles = @()

# Get all Python files recursively
$pythonFiles = Get-ChildItem -Path "emotion_net" -Filter "*.py" -Recurse
Write-Host "Found $($pythonFiles.Count) Python files to check" -ForegroundColor Cyan

foreach ($file in $pythonFiles) {
    Write-Host "Processing $($file.FullName)..." -ForegroundColor Gray
    try {
        # Read file as binary
        $content = [System.IO.File]::ReadAllBytes($file.FullName)
        
        # Check for null bytes
        $hasNullBytes = $content -contains 0
        
        # Check for CRLF
        $hasCRLF = [System.Text.Encoding]::UTF8.GetString($content).Contains("`r`n")
        
        if ($hasNullBytes) {
            Write-Host "  - Contains null bytes" -ForegroundColor Red
        }
        
        if ($hasCRLF) {
            Write-Host "  - Contains Windows line endings (CRLF)" -ForegroundColor Yellow
        }
        
        # If has null bytes or uses CRLF, fix it
        if ($hasNullBytes -or $hasCRLF) {
            Write-Host "  - Fixing file: $($file.FullName)" -ForegroundColor Yellow
            
            # Convert to string, replace CRLF with LF
            $text = [System.Text.Encoding]::UTF8.GetString($content)
            $text = $text -replace "`r`n", "`n"
            
            # Remove null bytes if any
            if ($hasNullBytes) {
                $text = $text -replace "`0", ""
                Write-Host "  - Removed null bytes" -ForegroundColor Green
            }
            
            # Write back as UTF8
            [System.IO.File]::WriteAllText($file.FullName, $text, [System.Text.Encoding]::UTF8)
            Write-Host "  - File fixed" -ForegroundColor Green
            
            $count++
            $fixedFiles += $file.FullName
        } else {
            Write-Host "  - No issues found" -ForegroundColor Green
        }
    }
    catch {
        Write-Host "Error processing $($file.FullName): $_" -ForegroundColor Red
    }
}

Write-Host "Fixed $count files for Linux compatibility" -ForegroundColor Green

if ($count -gt 0) {
    Write-Host "Fixed files:" -ForegroundColor Cyan
    $fixedFiles | ForEach-Object { Write-Host "  $_" }
} 