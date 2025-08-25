#!/usr/bin/env python3

import argparse
import sys

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Convert node file to YAML tag format')
    parser.add_argument('--input-file', required=True, help='Path to the input txt file')
    parser.add_argument('--connection-id', required=True, help='Connection ID string')
    
    args = parser.parse_args()
    
    try:
        # Read the input file
        with open(args.input_file, 'r') as file:
            lines = file.readlines()
        
        # Print the tags header
        print("  tags:")
        
        # Process each line
        for line in lines:
            line = line.strip()
            if not line:  # Skip empty lines
                continue
                
            # Split by tab to get node_id and node_name
            parts = line.split('\t')
            if len(parts) >= 2:
                node_id = parts[0]
                node_name = parts[1]
                
                # Print the YAML formatted output
                print(f"  - name: {node_name}")
                print(f"    connection_id: {args.connection_id}")
                print(f'    node_identifier: "{node_id}"')
                print()
    
    except FileNotFoundError:
        print(f"Error: File '{args.input_file}' not found.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
