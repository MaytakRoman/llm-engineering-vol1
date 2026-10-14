def kv_bytes(layers, batch, tokens, heads, dim, bytes_):
    return 2 * layers * batch * tokens * heads * dim * bytes_

for heads in (32, 8, 1):
    size = kv_bytes(32, 1, 32768, heads, 128, 2)
    print("KV heads", heads, "GiB", size / 2**30)
assert kv_bytes(32, 1, 32768, 1, 128, 2) == 2**29
