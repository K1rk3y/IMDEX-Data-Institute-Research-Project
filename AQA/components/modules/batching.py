import numpy as np


def _batch_by_size_fn(indices, num_tokens_fn, max_tokens, max_sentences, req_mult):
    """
    Batches indices by accumulating examples until either:
      - (current batch size + 1) * (max token length in batch) > max_tokens, or
      - (current batch size + 1) > max_sentences.
    If req_mult > 1 the final batch is dropped if its size is not a multiple.
    """
    batches = []
    cur_batch = []
    cur_max = 0
    for idx in indices:
        nt = num_tokens_fn(idx)
        new_max = max(cur_max, nt)
        if cur_batch:
            if (max_tokens != -1 and (len(cur_batch) + 1) * new_max > max_tokens) or \
               (max_sentences != -1 and len(cur_batch) + 1 > max_sentences):
                batches.append(cur_batch)
                cur_batch = []
                cur_max = 0
        cur_batch.append(idx)
        cur_max = max(cur_max, nt)
    if cur_batch:
        batches.append(cur_batch)
    # Enforce required batch size multiple by dropping the final batch if needed.
    if req_mult > 1 and len(batches[-1]) % req_mult != 0:
        batches = batches[:-1]
    return batches


def _batch_by_size_vec(indices, num_tokens_vec, max_tokens, max_sentences, req_mult):
    """
    Similar to _batch_by_size_fn but uses a precomputed vector of token counts.
    """
    batches = []
    cur_batch = []
    cur_max = 0
    for idx in indices:
        nt = int(num_tokens_vec[idx])
        new_max = max(cur_max, nt)
        if cur_batch:
            if (max_tokens != -1 and (len(cur_batch) + 1) * new_max > max_tokens) or \
               (max_sentences != -1 and len(cur_batch) + 1 > max_sentences):
                batches.append(cur_batch)
                cur_batch = []
                cur_max = 0
        cur_batch.append(idx)
        cur_max = max(cur_max, nt)
    if cur_batch:
        batches.append(cur_batch)
    if req_mult > 1 and len(batches[-1]) % req_mult != 0:
        batches = batches[:-1]
    return batches


def _batch_fixed_shapes_fast(indices, num_tokens_fn, fixed_shapes_sorted):
    """
    Given fixed shapes (each a tuple (batch_size, max_tokens)), partition the indices
    in order. (Here we assume that the fixed shape already comes from a length-bucketing
    routine and we simply split the indices accordingly.)
    """
    batches = []
    indices = list(indices)
    pos = 0
    for shape in fixed_shapes_sorted:
        batch_size, max_seq_tokens = shape
        if pos + batch_size > len(indices):
            break
        # Optionally you could check that max(num_tokens_fn(idx) for idx in batch)
        # is <= max_seq_tokens.
        batch = indices[pos:pos+batch_size]
        batches.append(batch)
        pos += batch_size
    return batches


def batch_by_size(
    indices,
    num_tokens_fn,
    num_tokens_vec=None,
    max_tokens=None,
    max_sentences=None,
    required_batch_size_multiple=1,
    fixed_shapes=None,
):
    """
    Returns batches of indices such that each batch (roughly) satisfies:
      - batch_size * (max token length in batch) <= max_tokens, and
      - batch_size <= max_sentences.
    If fixed_shapes is provided (a list of tuples (batch_size, max_tokens)), then those
    shapes are used (and max_sentences/required_batch_size_multiple are ignored).
    
    Args:
        indices (iterable of int): ordered list of dataset indices.
        num_tokens_fn (callable): function mapping an index to its token count.
        num_tokens_vec (iterable of int, optional): precomputed token counts.
        max_tokens (int, optional): maximum tokens allowed in a batch.
        max_sentences (int, optional): maximum sentences allowed in a batch.
        required_batch_size_multiple (int, optional): if >1, final batch must have a size
            that is a multiple of this number (otherwise it is dropped).
        fixed_shapes (list of (int, int), optional): if given, batching will partition the
            indices according to these shapes.
    """
    max_tokens = int(max_tokens) if max_tokens is not None else -1
    max_sentences = int(max_sentences) if max_sentences is not None else -1
    req_mult = required_batch_size_multiple

    # Ensure indices is a NumPy array of int64.
    if not isinstance(indices, np.ndarray):
        indices = np.array(list(indices), dtype=np.int64)

    if num_tokens_vec is not None and not isinstance(num_tokens_vec, np.ndarray):
        num_tokens_vec = np.array(list(num_tokens_vec), dtype=np.int64)

    if fixed_shapes is None:
        if num_tokens_vec is None:
            batches = _batch_by_size_fn(indices, num_tokens_fn, max_tokens, max_sentences, req_mult)
        else:
            batches = _batch_by_size_vec(indices, num_tokens_vec, max_tokens, max_sentences, req_mult)
        return batches
    else:
        fixed_shapes = np.array(fixed_shapes, dtype=np.int64)
        # Sort fixed_shapes by batch size then token length.
        sort_order = np.lexsort((fixed_shapes[:, 1], fixed_shapes[:, 0]))
        fixed_shapes_sorted = fixed_shapes[sort_order]
        return _batch_fixed_shapes_fast(indices, num_tokens_fn, fixed_shapes_sorted)


# A version that mimics a method which uses externally provided batch shapes.
def batch_by_size_with_shapes(
    indices,
    num_tokens_fn,
    get_batch_shapes_fn,
    num_tokens_vec_fn=None,
    max_tokens=None,
    max_sentences=None,
    required_batch_size_multiple=1,
):
    """
    Given an ordered set of indices, this version obtains fixed batch shapes from
    get_batch_shapes_fn. If fixed shapes are available, we adjust each shape based on
    max_tokens/max_sentences and then use them for batching.
    
    Args:
        get_batch_shapes_fn (callable): returns a list of (batch_size, num_tokens) or None.
        num_tokens_vec_fn (callable, optional): returns token counts for indices.
    """
    fixed_shapes = get_batch_shapes_fn()
    if fixed_shapes is not None:
        def adjust_bsz(bsz, num_tokens):
            if bsz is None:
                assert max_tokens is not None, "Must specify max_tokens if bsz is not provided"
                bsz = max_tokens // num_tokens
            if max_sentences is not None:
                bsz = min(bsz, max_sentences)
            elif bsz >= required_batch_size_multiple and bsz % required_batch_size_multiple != 0:
                bsz -= bsz % required_batch_size_multiple
            return bsz

        fixed_shapes = np.array(
            [[adjust_bsz(bsz, num_tokens), num_tokens] for (bsz, num_tokens) in fixed_shapes],
            dtype=np.int64,
        )
    if num_tokens_vec_fn is not None:
        try:
            num_tokens_vec = num_tokens_vec_fn(indices).astype("int64")
        except NotImplementedError:
            num_tokens_vec = None
    else:
        num_tokens_vec = None

    return batch_by_size(
        indices,
        num_tokens_fn,
        num_tokens_vec=num_tokens_vec,
        max_tokens=max_tokens,
        max_sentences=max_sentences,
        required_batch_size_multiple=required_batch_size_multiple,
        fixed_shapes=fixed_shapes,
    )