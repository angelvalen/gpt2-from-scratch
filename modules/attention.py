import torch

from einops import rearrange
from torch import nn


class CausalSelfAttention(nn.Module):
  def __init__(self, config):
    super().__init__()

    self.num_attention_heads = config.num_attention_heads
    self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
    self.all_head_size = self.num_attention_heads * self.attention_head_size

    # Initialize the linear transformation layers for key, value, query.
    self.query = nn.Linear(config.hidden_size, self.all_head_size)
    self.key = nn.Linear(config.hidden_size, self.all_head_size)
    self.value = nn.Linear(config.hidden_size, self.all_head_size)
    # This dropout is applied to normalized attention scores following the original
    # implementation of transformer. Although it is a bit unusual, we empirically
    # observe that it yields better performance.
    self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

  def transform(self, x, linear_layer):
    # The corresponding linear_layer of k, v, q are used to project the hidden_state (x).
    proj = linear_layer(x)
    # Next, we need to produce multiple heads for the proj. This is done by spliting the
    # hidden state to self.num_attention_heads, each of size self.attention_head_size.
    proj = rearrange(proj, 'b t (h d) -> b t h d', h=self.num_attention_heads)
    # By proper transpose, we have proj of size [bs, num_attention_heads, seq_len, attention_head_size].
    proj = rearrange(proj, 'b t h d -> b h t d')
    return proj

  def attention(self, key, query, value, attention_mask):

    ### YOUR CODE HERE
    d_k = self.attention_head_size
    att_score = torch.matmul(query, key.transpose(-1, -2)) / torch.sqrt(torch.tensor(d_k, dtype=float, device=key.device))

    q_len = query.shape[-2]
    k_len = key.shape[-2]
    causal_mask = - torch.triu(torch.ones(q_len, k_len, device=key.device), diagonal=k_len - q_len + 1) * 10000.0 
    att_masked = att_score + causal_mask + attention_mask # Causal and padding mask

    att_probs = self.dropout(torch.softmax(att_masked, dim=-1)) # Dropout to probs, as in original GPT2
    output = torch.matmul(att_probs, value)
    output_concat = rearrange(output, 'b h t d -> b t (h d)')
    return output_concat

  def forward(self, hidden_states, attention_mask, layer_kv_cache=None):
    """
    hidden_states: [bs, seq_len, hidden_state] ([bs, 1, hidden] if kv cache is given)
    attention_mask: [bs, 1, 1, seq_len]
    output: [bs, seq_len, hidden_state] ([bs, 1, hidden] if kv cache is given)
    """
    # First, we have to generate the key, value, query for each token for multi-head attention
    # using self.transform (more details inside the function).
    # Size of *_layer is [bs, num_attention_heads, seq_len, attention_head_size].
    if layer_kv_cache is None:
      key_layer = self.transform(hidden_states, self.key)
      value_layer = self.transform(hidden_states, self.value)
      query_layer = self.transform(hidden_states, self.query)
      
      # Calculate the multi-head attention.
      output = self.attention(key_layer, query_layer, value_layer, attention_mask)

      new_layer_kv_cache = (key_layer, value_layer)

      return output, new_layer_kv_cache
    
    else:
      cached_keys = layer_kv_cache[0] # [bs, h, seq_len - 1, d/h]
      last_key = self.transform(hidden_states, self.key) # [bs, h, 1, d/h]
      keys = torch.cat((cached_keys, last_key), dim=-2)

      cached_values = layer_kv_cache[1] # [bs, h, seq_len - 1, d/h]
      last_value = self.transform(hidden_states, self.value) # [bs, h, 1, d/h]
      values = torch.cat((cached_values, last_value), dim=-2)

      last_query = self.transform(hidden_states, self.query)

      output = self.attention(keys, last_query, values, attention_mask) 
      # [bs, 1, d], so now seq_len = 1, but generation() code will take the last token anyway, so previous arent needed

      new_layer_kv_cache = (keys, values)

      return output, new_layer_kv_cache
