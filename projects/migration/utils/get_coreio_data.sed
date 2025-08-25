# NOTE: run with -E flag

# first, combine info in id, process, name 
# to construct tag name for PLC tags
# NOTE: for some reason, we shouldn't escape
# the closing bracket in [^]] (any character except closing bracket)
# s/ns=1;s=\[.+-([^]]+)\][^\t]+\t([^\t]+)\t([^\t]+)\t/\L\1_\2_\3\t/
s/(ns=1;s=\[.+-)([^]]+)(\][^\t]+\t)([^\t]+)\t([^\t]+)\t/\1\2\3\L\2_\4_\5\t/


# add rlcore label to workaround tags from custom micro OPC
# and construct tag name
s/(ns=1;[^\t]+\t)([^\t]+)\t([^\t]+)\t/\1\Lrlcore_\2_\3\t/

# clean up double uf1_uf1 or uf2_uf2
s/uf([12])_uf\1/uf\1/

# grep -P '^[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}\+00\t' clean_tail.txt
# grep -E '^[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}\+[0-9]{2}' clean_tail.txt

# drop dtype
s/([^\t]+\t[^\t]+)\t[^\t]+/\1/
