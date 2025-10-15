# BPE example

text = "low low low low low lower lower widest widest widest newest newest newest newest newest newest"

# Init vocab

# split on spaces
words = text.split()
word_count = {}
for w in words:
    word_count[w] = word_count.get(w, 0) + 1

print(word_count)

byte_tuple_count = {
    tuple(ch.encode("utf-8") for ch in word): count for word, count in word_count.items()
}

print("Byte Tuple Count:")
print(byte_tuple_count)

# Merge
for i in range(6):
    print("iteration #", i+1)
    byte_pair_count = {}
    for tup, count in byte_tuple_count.items():
        pairs = [(tup[i], tup[i+1]) for i in range(len(tup)-1)]

        for p in pairs:
            byte_pair_count[p] = byte_pair_count.get(p, 0) + count

    print("Byte Pair Count:")
    print(byte_pair_count)
    largest_pair_count = max(byte_pair_count.items(), key=lambda x:(x[1],x[0]))
    print("pair to merge:")
    print(largest_pair_count[0])

    merged_tuple_count = {}
    first, second = largest_pair_count[0]
    merged_byte = b"".join(largest_pair_count[0])
    for tup, count in byte_tuple_count.items():
        new_tuple = []
        i = 0
        while i < len(tup):
            if i < len(tup) - 1 and tup[i] == first and tup[i + 1] == second:
                new_tuple.append(merged_byte)
                i += 2
            else:
                new_tuple.append(tup[i])
                i += 1
        new_tuple = tuple(new_tuple)
        merged_tuple_count[new_tuple] = merged_tuple_count.get(new_tuple, 0) + count
    byte_tuple_count = merged_tuple_count
    print("After Merge:")
    print(byte_tuple_count)