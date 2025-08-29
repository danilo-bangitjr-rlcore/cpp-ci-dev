/"/!d # if line doesn't have a quote, skip it
/"time"/d # skip time
s/^(".+").*/\1,/ # remove dtype, just get name followed by comma
