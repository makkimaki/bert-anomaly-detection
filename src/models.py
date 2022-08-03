from turtle import forward, position
from torch import nn
import torch


class BertLayerNorm(nn.Module):
    """LayerNormalization
    """

    def __init__(self, hidden_size, eps=1e-12) -> None:
        super(BertLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))  # weight
        self.beta = nn.Parameter(torch.zeros(hidden_size))  # bias
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta
        
class BertEmbeddings(nn.Module):
    """
    Obtain embedding vector from 
       1. word ID series of sentences
       2. the descriptor which distinguishes the 1st and 2nd sentences
    """

    def __init__(self, config) -> None:
        super(BertEmbeddings).__init__()

        # Earn the 3 kinds vector embeddings

        self.word_embeddings = nn.Embedding(
            config.vocab_size, 
            config.hidden_size,
            padding_idx=0
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, 
            config.hidden_size
        )
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, 
            config.hidden_size
        )
        self.LayerNorm = BertLayerNorm(
            config.hidden_size,
            eps=1e-12
        )
        self.dropout = nn.Dropout(
            config.hidden_dropout_prob
        )
    
    def forward(self, input_ids, token_type_ids=None):
        """
        Args: 
            input_ids:
            token_type_ids:  

        """
        # 1. token embeddings
        words_embeddings = self.word_embeddings(input_ids)

        # 2. sentence embeddings
        # if no token_type_ids, 。。。？
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # 3. Transformer Positional Embedding:
        seq_length = input_ids.size(1)
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=input_ids.device
            )
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        # sum up the 3 embedding tensors [batch_size, seq_len, hidden_size]
        embeddings = words_embeddings + position_embeddings + token_type_embeddings

        # Execute LayerNormalization and Dropout
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class BertLayer(nn.Module):
    """ Transformer
    """

    def __init__(self, config) -> None:
        super(BertLayer).__init__()

        # self-attention
        self.attention = BertAttention(config)

        # FC layer that process the self-attention output
        self.intermediate = BertIntermediate(config)

        # Last layer that sum up features from self-attention and input for BertLayer
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask, attention_show_fig=False):
        """
        Args: 
            hidden_states: 
            attentipon_mask: 
            attention_show_fig: 

        Returns: 
            layer_output:
            attention_probs: 
        """
        if attention_show_fig:
            """"""
            attention_output, attention_probs = self.attention(
                hidden_states, attention_mask, attention_show_fig
            )
            intermediate_output = self.intermediate(attention_output)
            layer_output = self.output(intermediate_output, attention_output)

            return layer_output, attention_probs
        
        elif attention_show_fig == False:
            attention_output = self.attention(
                hidden_states, attention_mask, attention_show_fig
            )
            intermediate_output = self.intermediate(attention_output)
            layer_output = self.output(intermediate_output, attention_output)

            return layer_output  # [batch_size, seq_length, hidden_size]
            

    class BertAttention(nn.Module):
        """ Self-Attention part in BertLayer module.
        """

        def __init__(self, config) -> None:
            super(BertAttention).__init__()
            self.selfattn = BertSelfAttention(config)
            self.output = BertSelfOutput(config)

        def forward(self, input_tensor, attention_mask, attention_show_flg=False):
            """

            Args: 
                input_tensor: 
                attention_mask: 
                attention_show_flg:
            """
            if attention_show_flg:
                # This returns attention_probs as well.
                self_output, attention_probs = self.selfattn(input_tensor, attention_mask, attention_show_flg)
                attention_output = self.output(self_output, input_tensor)
                return attention_output, attention_probs

            elif attention_show_flg == False:
                self_output = self.selfattn(input_tensor, attention_mask, attention_show_flg)
                attention_output = self.output(self_output, input_tensor)
                return attention_output
