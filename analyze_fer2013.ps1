# FER2013 Dataset Analysis Script
# This script analyzes the structure and content of the FER2013 dataset

# Display dataset files information
Write-Host "=== FER2013 Dataset Files ===" -ForegroundColor Green
Get-ChildItem -Path "data/fer2013" | Format-Table Name, Length, @{Name="Size(MB)";Expression={"{0:N2}" -f ($_.Length / 1MB)}}

# Emotion labels in FER2013
$emotionLabels = @{
    0 = "Angry"
    1 = "Disgust" 
    2 = "Fear"
    3 = "Happy"
    4 = "Sad"
    5 = "Surprise"
    6 = "Neutral"
}

# Function to analyze CSV file
function Analyze-FER2013File {
    param(
        [string]$filePath,
        [string]$fileType
    )
    
    Write-Host "`n=== Analyzing $fileType Dataset ===" -ForegroundColor Green
    
    try {
        # Check if file exists
        if (-not (Test-Path $filePath)) {
            Write-Host "File not found: $filePath" -ForegroundColor Red
            return
        }
        
        # Get header info
        $headers = Get-Content $filePath -TotalCount 1
        Write-Host "Headers: $headers"
        
        # Count rows (minus header)
        $lineCount = (Get-Content $filePath | Measure-Object -Line).Lines - 1
        Write-Host "Number of samples: $lineCount"
        
        # Sample a few rows
        Write-Host "`nSample entries (first 3):"
        Get-Content $filePath -TotalCount 4 | Select-Object -Skip 1 | ForEach-Object {
            $line = $_
            if ($line -match "^(\d),") {
                $emotion = $matches[1]
                Write-Host "  Emotion: $emotion ($($emotionLabels[$emotion]))"
            }
            elseif ($fileType -eq "Test" -and $line -match '^"(.+)"$') {
                # Handle test file format (just pixels, no emotion label)
                Write-Host "  Pixels data (first 20 values): $($matches[1].Substring(0, 60))..."
            }
        }
    }
    catch {
        Write-Host "Error analyzing file: $_" -ForegroundColor Red
    }
}

# Check the structure of icml_face_data.csv file
function Analyze-FullDataset {
    param([string]$filePath)
    
    Write-Host "`n=== Analyzing Full Dataset File ===" -ForegroundColor Green
    
    try {
        # Check if file exists
        if (-not (Test-Path $filePath)) {
            Write-Host "File not found: $filePath" -ForegroundColor Red
            return
        }
        
        # Get header info
        $headers = Get-Content $filePath -TotalCount 1
        Write-Host "Headers: $headers"
        
        # Count rows (minus header)
        $lineCount = (Get-Content $filePath | Measure-Object -Line).Lines - 1
        Write-Host "Number of samples: $lineCount"
        
        # Sample rows
        Write-Host "`nSample entries (first 3):"
        Get-Content $filePath -TotalCount 4 | Select-Object -Skip 1 | ForEach-Object {
            $line = $_
            if ($line -match '^(\d),"(.+)",(\w+)$') {
                $emotion = $matches[1]
                $pixelsStart = $matches[2].Substring(0, 60)
                $usage = $matches[3]
                Write-Host "  Emotion: $emotion ($($emotionLabels[$emotion])), Usage: $usage, Pixels (start): $pixelsStart..."
            }
        }
    } catch {
        Write-Host "Error analyzing full dataset: $_" -ForegroundColor Red
    }
}

# Function to analyze emotion distribution in CSV file
function Analyze-EmotionDistribution {
    param(
        [string]$filePath,
        [string]$fileType
    )
    
    Write-Host "`n=== Emotion Distribution in $fileType Dataset ===" -ForegroundColor Green
    
    try {
        # Check if file has emotion labels (test file might not)
        $header = Get-Content $filePath -TotalCount 1
        if (-not $header.Contains("emotion")) {
            Write-Host "This file doesn't contain emotion labels." -ForegroundColor Yellow
            return
        }
        
        # Check file size first
        $fileSize = (Get-Item $filePath).Length / 1MB
        
        # If file is too large, read only first 1000 lines
        if ($fileSize -gt 100) {
            Write-Host "File is large ($($fileSize.ToString("N2")) MB), sampling first 1000 entries" -ForegroundColor Yellow
            $data = Get-Content $filePath -TotalCount 1001 | Select-Object -Skip 1
        }
        else {
            $data = Get-Content $filePath | Select-Object -Skip 1
        }
        
        # Count emotions
        $emotionCounts = @{}
        foreach ($emotion in 0..6) {
            $emotionCounts[$emotion] = 0
        }
        
        foreach ($line in $data) {
            if ($line -match "^(\d),") {
                $emotion = [int]$matches[1]
                $emotionCounts[$emotion]++
            }
        }
        
        # Display counts
        Write-Host "`nEmotion distribution (from sample):"
        foreach ($emotion in 0..6) {
            $count = $emotionCounts[$emotion]
            $percentage = if ($data.Count -gt 0) { [math]::Round(($count / $data.Count) * 100, 2) } else { 0 }
            Write-Host "  $($emotionLabels[$emotion]): $count ($percentage%)"
        }
    }
    catch {
        Write-Host "Error analyzing emotion distribution: $_" -ForegroundColor Red
    }
}

# Check what the example_submission.csv looks like
function Check-SubmissionFormat {
    param([string]$filePath)
    
    Write-Host "`n=== Checking Submission Format ===" -ForegroundColor Green
    
    try {
        # Check if file exists
        if (-not (Test-Path $filePath)) {
            Write-Host "File not found: $filePath" -ForegroundColor Red
            return
        }
        
        # Count rows
        $lineCount = (Get-Content $filePath | Measure-Object -Line).Lines
        Write-Host "Number of lines: $lineCount"
        
        # Sample rows
        Write-Host "`nSample entries (first 10):"
        Get-Content $filePath -TotalCount 10 | ForEach-Object {
            Write-Host "  $_"
        }
        
        # Try to understand format
        Write-Host "`nFormat analysis:"
        $allNumbers = $true
        $sample = Get-Content $filePath -TotalCount 20
        foreach ($line in $sample) {
            if (-not [int]::TryParse($line, [ref]$null)) {
                $allNumbers = $false
                break
            }
        }
        
        if ($allNumbers) {
            Write-Host "  The submission file appears to contain emotion class predictions (0-6)" -ForegroundColor Green
            
            # Count distribution 
            $emotionCounts = @{}
            foreach ($emotion in 0..6) {
                $emotionCounts[$emotion] = 0
            }
            
            $allLines = Get-Content $filePath
            foreach ($line in $allLines) {
                $emotion = [int]$line
                if ($emotionCounts.ContainsKey($emotion)) {
                    $emotionCounts[$emotion]++
                }
            }
            
            Write-Host "`nEmotion distribution in submission:"
            foreach ($emotion in 0..6) {
                $count = $emotionCounts[$emotion]
                $percentage = if ($allLines.Count -gt 0) { [math]::Round(($count / $allLines.Count) * 100, 2) } else { 0 }
                Write-Host "  $($emotionLabels[$emotion]): $count ($percentage%)"
            }
        }
        else {
            Write-Host "  The submission file format could not be determined" -ForegroundColor Yellow
        }
    } catch {
        Write-Host "Error checking submission format: $_" -ForegroundColor Red
    }
}

# Analyze dataset files
Analyze-FER2013File "data/fer2013/train.csv" "Training"
Analyze-EmotionDistribution "data/fer2013/train.csv" "Training"

Analyze-FER2013File "data/fer2013/test.csv" "Test"

# Check the main dataset file and submission format
if (Test-Path "data/fer2013/icml_face_data.csv") {
    Analyze-FullDataset "data/fer2013/icml_face_data.csv"
}

Check-SubmissionFormat "data/fer2013/example_submission.csv"

# Display additional dataset information
Write-Host "`n=== FER2013 Dataset Information ===" -ForegroundColor Green
Write-Host "- FER2013 is a facial expression recognition dataset"
Write-Host "- It contains grayscale images of faces showing different emotions"
Write-Host "- Images are 48x48 pixels"
Write-Host "- Each image is labeled with one of 7 emotion categories:"
foreach ($e in 0..6) {
    Write-Host "  $e`: $($emotionLabels[$e])"
}
Write-Host "- The dataset was created for the ICML 2013 Workshop challenge"
Write-Host "- The 'pixels' column contains the flattened 48x48 image (2304 values)"
Write-Host "- Training set contains labeled emotions, test set contains only pixel data"
Write-Host "- The full dataset (icml_face_data.csv) includes a 'Usage' column indicating"
Write-Host "  which samples belong to Training, PublicTest, or PrivateTest sets" 