from transformers.utils import ModelOutput
import torch
import typing as t


def _flatten_and_batch_shift_indices(
    indices: torch.Tensor, sequence_length: int
) -> torch.Tensor:
    """
    This is a subroutine for [`batched_index_select`](./util.md#batched_index_select).
    The given `indices` of size `(batch_size, d_1, ..., d_n)` indexes into dimension 2 of a
    target tensor, which has size `(batch_size, sequence_length, embedding_size)`. This
    function returns a vector that correctly indexes into the flattened target. The sequence
    length of the target must be provided to compute the appropriate offsets.

    ```python
        indices = torch.ones([2,3], dtype=torch.long)
        # Sequence length of the target tensor.
        sequence_length = 10
        shifted_indices = flatten_and_batch_shift_indices(indices, sequence_length)
        # Indices into the second element in the batch are correctly shifted
        # to take into account that the target tensor will be flattened before
        # the indices are applied.
        assert shifted_indices == [1, 1, 1, 11, 11, 11]
    ```

    # Parameters

    indices : `torch.LongTensor`, required.
    sequence_length : `int`, required.
        The length of the sequence the indices index into.
        This must be the second dimension of the tensor.

    # Returns

    offset_indices : `torch.LongTensor`
    """
    # Shape: (batch_size)
    if torch.max(indices) >= sequence_length or torch.min(indices) < 0:
        raise f"All elements in indices should be in range (0, {sequence_length - 1})"

    device = indices.get_device() if indices.is_cuda else -1
    offsets = _get_range_vector(indices.size(0), device) * sequence_length
    for _ in range(len(indices.size()) - 1):
        offsets = offsets.unsqueeze(1)

    # Shape: (batch_size, d_1, ..., d_n)
    offset_indices = indices + offsets

    # Shape: (batch_size * d_1 * ... * d_n)
    offset_indices = offset_indices.view(-1)
    return offset_indices


def batched_index_select(
    target: torch.Tensor,
    indices: torch.LongTensor,
    flattened_indices: t.Optional[torch.LongTensor] = None,
) -> torch.Tensor:
    """
    The given `indices` of size `(batch_size, d_1, ..., d_n)` indexes into the sequence
    dimension (dimension 2) of the target, which has size `(batch_size, sequence_length,
    embedding_size)`.

    This function returns selected values in the target with respect to the provided indices, which
    have size `(batch_size, d_1, ..., d_n, embedding_size)`. This can use the optionally
    precomputed `flattened_indices` with size `(batch_size * d_1 * ... * d_n)` if given.

    An example use case of this function is looking up the start and end indices of spans in a
    sequence tensor. This is used in the
    [CoreferenceResolver](https://docs.allennlp.org/models/main/models/coref/models/coref/)
    model to select contextual word representations corresponding to the start and end indices of
    mentions.

    The key reason this can't be done with basic torch functions is that we want to be able to use look-up
    tensors with an arbitrary number of dimensions (for example, in the coref model, we don't know
    a-priori how many spans we are looking up).

    # Parameters

    target : `torch.Tensor`, required.
        A 3 dimensional tensor of shape (batch_size, sequence_length, embedding_size).
        This is the tensor to be indexed.
    indices : `torch.LongTensor`
        A tensor of shape (batch_size, ...), where each element is an index into the
        `sequence_length` dimension of the `target` tensor.
    flattened_indices : `Optional[torch.Tensor]`, optional (default = `None`)
        An optional tensor representing the result of calling `flatten_and_batch_shift_indices`
        on `indices`. This is helpful in the case that the indices can be flattened once and
        cached for many batch lookups.

    # Returns

    selected_targets : `torch.Tensor`
        A tensor with shape [indices.size(), target.size(-1)] representing the embedded indices
        extracted from the batch flattened target tensor.
    """
    if flattened_indices is None:
        # Shape: (batch_size * d_1 * ... * d_n)
        flattened_indices = _flatten_and_batch_shift_indices(indices, target.size(1))

    # Shape: (batch_size * sequence_length, embedding_size)
    flattened_target = target.view(-1, target.size(-1))

    # Shape: (batch_size * d_1 * ... * d_n, embedding_size)
    flattened_selected = flattened_target.index_select(0, flattened_indices)
    selected_shape = list(indices.size()) + [target.size(-1)]
    # Shape: (batch_size, d_1, ..., d_n, embedding_size)
    selected_targets = flattened_selected.view(*selected_shape)
    return selected_targets


def _get_range_vector(size: int, device: int) -> torch.Tensor:
    """
    Returns a range vector with the desired size, starting at 0. The CUDA implementation
    is meant to avoid copy data from CPU to GPU.
    """
    if device > -1:
        return torch.cuda.LongTensor(size, device=device).fill_(1).cumsum(0) - 1
    else:
        return torch.arange(0, size, dtype=torch.long)


def _batched_span_select(
    target: torch.Tensor, spans: torch.LongTensor
) -> t.Tuple[torch.Tensor, torch.BoolTensor]:
    """
    The given `spans` of size `(batch_size, num_spans, 2)` indexes into the sequence
    dimension (dimension 2) of the target, which has size `(batch_size, sequence_length,
    embedding_size)`.

    This function returns segmented spans in the target with respect to the provided span indices.

    # Parameters

    target : `torch.Tensor`, required.
        A 3 dimensional tensor of shape (batch_size, sequence_length, embedding_size).
        This is the tensor to be indexed.
    indices : `torch.LongTensor`
        A 3 dimensional tensor of shape (batch_size, num_spans, 2) representing start and end
        indices (both inclusive) into the `sequence_length` dimension of the `target` tensor.

    # Returns

    span_embeddings : `torch.Tensor`
        A tensor with shape (batch_size, num_spans, max_batch_span_width, embedding_size]
        representing the embedded spans extracted from the batch flattened target tensor.
    span_mask: `torch.BoolTensor`
        A tensor with shape (batch_size, num_spans, max_batch_span_width) representing the mask on
        the returned span embeddings.
    """
    # both of shape (batch_size, num_spans, 1)
    span_starts, span_ends = spans.split(1, dim=-1)

    # shape (batch_size, num_spans, 1)
    # These span widths are off by 1, because the span ends are `inclusive`.
    span_widths = span_ends - span_starts

    # We need to know the maximum span width so we can
    # generate indices to extract the spans from the sequence tensor.
    # These indices will then get masked below, such that if the length
    # of a given span is smaller than the max, the rest of the values
    # are masked.
    max_batch_span_width = span_widths.max().item() + 1

    # Shape: (1, 1, max_batch_span_width)
    device = target.get_device() if target.is_cuda else -1
    max_span_range_indices = _get_range_vector(max_batch_span_width, device).view(
        1, 1, -1
    )
    # Shape: (batch_size, num_spans, max_batch_span_width)
    # This is a broadcasted comparison - for each span we are considering,
    # we are creating a range vector of size max_span_width, but masking values
    # which are greater than the actual length of the span.
    #
    # We're using <= here (and for the mask below) because the span ends are
    # inclusive, so we want to include indices which are equal to span_widths rather
    # than using it as a non-inclusive upper bound.
    span_mask = max_span_range_indices <= span_widths
    raw_span_indices = span_starts + max_span_range_indices
    # We also don't want to include span indices which greater than the sequence_length,
    # which happens because some spans near the end of the sequence
    # have a start index + max_batch_span_width > sequence_length, so we add this to the mask here.
    span_mask = (
        span_mask & (raw_span_indices < target.size(1)) & (0 <= raw_span_indices)
    )
    span_indices = raw_span_indices * span_mask

    # Shape: (batch_size, num_spans, max_batch_span_width, embedding_dim)
    span_embeddings = batched_index_select(target, span_indices)

    return span_embeddings, span_mask


def realign_embeddings(
    model_outputs: ModelOutput, offsets: torch.Tensor
) -> torch.Tensor:
    """
    Use the bert module to calculate embedding and apply postprocessing steps.

    the post processing masking and reshaping is taken from the allennlp embedder, to make the output shape
    similar to the one returned by the original model.
    Original code can be found here:
    https://github.com/allenai/allennlp/blob/main/allennlp/modules/token_embedders/pretrained_transformer_mismatched_embedder.py

    Parameters
    ----------
    model_outputs : transformers.utils.ModelOutput
        Output from a huggingface transformer model
    offsets: torch.LongTensor
        The offsets of input words in the tokenization output

    Returns
    -------
    Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]
        embedding output
    """
    span_embeddings, span_mask = _batched_span_select(
        model_outputs.last_hidden_state, offsets
    )
    span_mask = span_mask.unsqueeze(-1)
    span_embeddings *= span_mask
    # Sum over embeddings of all sub-tokens of a word
    span_embeddings_sum = span_embeddings.sum(2)
    # Shape (batch_size, num_orig_tokens)
    span_embeddings_len = span_mask.sum(2)
    # Find the average of sub-tokens embeddings by dividing `span_embedding_sum` by `span_embedding_len`
    # Shape: (batch_size, num_orig_tokens, embedding_size)
    orig_embeddings = span_embeddings_sum / torch.clamp_min(span_embeddings_len, 1)
    # All the places where the span length is zero, write in zeros.
    orig_embeddings[(span_embeddings_len == 0).expand(orig_embeddings.shape)] = 0

    return orig_embeddings


def get_offsets(text, tokenizer_output):
    """
    Map each word to the tokens representing it in the tokenized text.

    Parameters
    ----------
    text: str
        The original text that was tokenized by BertTokenizer.
    tokenizer_output: dict[str, Any]
        The output of BertTokenizer on the spacy tokenized text.

    Returns
    -------
    offsets: List[List[int]]
        The i'th item in the list is the offsets of the i'th word
        in the original text. the offset is a list of two int
        values: [start_token, end_token]
    """
    words = text.split()
    offsets = []

    token_index = (
        1  # start from 1 because the first token is the special sentence start token
    )
    for word in words:
        char_span = tokenizer_output.token_to_chars(token_index)
        token_text = text[char_span.start : char_span.end]
        start_index = token_index
        end_index = token_index
        # For a word that was split into more than 1 token,
        # keep adding tokens until the token text equals the word.
        while token_text != word:
            end_index += 1
            char_span = tokenizer_output.token_to_chars(end_index)
            token_text += text[char_span.start : char_span.end]

        offsets.append([start_index, end_index])
        token_index = end_index + 1

    return torch.LongTensor(offsets)


def reshape_tensor(input_tensor: torch.Tensor) -> torch.Tensor:
    input_size = input_tensor.size()
    if len(input_size) <= 2:
        raise RuntimeError(f"No dimension to distribute: {input_size}")
    # Squash batch_size and time_steps into a single axis; result has shape
    # (batch_size * time_steps, **input_size).
    squashed_shape = [-1] + list(input_size[2:])

    return input_tensor.contiguous().view(*squashed_shape)
