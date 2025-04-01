#!/bin/bash

# Check if a zip file is provided as an argument
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <zip_file>"
    exit 1
fi

ZIP_FILE=$1
EXTRACTED_FOLDER="extracted"
OUTPUT_FOLDER="processed"
RESULT_ZIP="results.zip"

echo "Extracting $ZIP_FILE..."
mkdir -p $EXTRACTED_FOLDER
unzip -q "$ZIP_FILE" -d $EXTRACTED_FOLDER

# Step 2: Install rembg with GPU CLI support
echo "Installing rembg[gpu,cli]..."
pip install "rembg[gpu,cli]"

# Step 3: Run rembg on the extracted folder
echo "Processing images with rembg..."
mkdir -p $OUTPUT_FOLDER
rembg p -m birefnet-portrait "$EXTRACTED_FOLDER" "$OUTPUT_FOLDER"

# Step 4: Compress the results into a new zip file
echo "Compressing results into $RESULT_ZIP..."
zip -r -q "$RESULT_ZIP" "$OUTPUT_FOLDER"

# Cleanup
# echo "Cleaning up temporary files..."
# rm -rf "$EXTRACTED_FOLDER" "$OUTPUT_FOLDER"

echo "Done! Results saved in $RESULT_ZIP."