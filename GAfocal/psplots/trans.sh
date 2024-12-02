#!/bin/bash

# Create the output directory if it doesn't exist
mkdir -p ./output

# Loop through all .ps files in the current directory
for ps_file in *.ps; do
    # Create the output file path in the ../output directory
    output_file="./output/${ps_file%.ps}.png"
    
    # Run the Ghostscript command
    gs -dSAFER -dBATCH -dNOPAUSE -sDEVICE=png16m -r300 -sOutputFile="$output_file" "$ps_file"
done
