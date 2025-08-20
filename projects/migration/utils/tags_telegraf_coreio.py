#!/usr/bin/env python3
import re
import argparse

def convert_opcua(input_file: str, connection_id: str):
    with open(input_file, 'r') as f:
        content = f.read()
    
    nodes = re.findall(r'\[\[inputs\.opcua\.nodes\]\].*?(?=\[\[|\Z)', content, re.DOTALL)
    
    print("  tags:")
    for node in nodes:
        ns_match = re.search(r'namespace = "([^"]*)"', node)
        id_type_match = re.search(r'identifier_type = "([^"]*)"', node)
        identifier_match = re.search(r'identifier = "([^"]*)"', node)
        name_match = re.search(r'name = "([^"]*)"', node.split('default_tags')[1])
        
        assert ns_match and id_type_match and identifier_match and name_match, f"Missing info at {node}"
        namespace = ns_match.group(1)
        id_type = id_type_match.group(1)
        identifier = identifier_match.group(1)
        name = name_match.group(1)
        
        node_id = f"ns={namespace};{id_type}={identifier}"
        print(f"  - name: {name}")
        print(f"    node_id: {node_id}")
        print(f"    connection_id: {connection_id}")
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', help='OPCUA configuration file')
    parser.add_argument('--connection-id', help='OPCUA config connection id', default="pilot_opc")
    args = parser.parse_args()
    convert_opcua(args.input_file, args.connection_id)
