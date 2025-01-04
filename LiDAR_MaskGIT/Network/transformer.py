# BERT architecture for the Masked Bidirectional Encoder Transformer
import torch
from torch import nn


class PreNorm(nn.Module):

    def __init__(self, dim, fn):
        """ PreNorm module to apply layer normalization before a given function
            :param:
                dim  -> int: Dimension of the input
                fn   -> nn.Module: The function to apply after layer normalization
            """
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        """ Forward pass through the PreNorm module
            :param:
                x        -> torch.Tensor: Input tensor
                **kwargs -> _ : Additional keyword arguments for the function
            :return
                torch.Tensor: Output of the function applied after layer normalization
        """
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        """ Initialize the Multi-Layer Perceptron (MLP).
            :param:
                dim        -> int : Dimension of the input
                dim        -> int : Dimension of the hidden layer
                dim        -> float : Dropout rate
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim, bias=True),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim, bias=True),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """ Forward pass through the MLP module.
            :param:
                x -> torch.Tensor: Input tensor
            :return
                torch.Tensor: Output of the function applied after layer
        """
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.):
        """ Initialize the Attention module.
            :param:
                embed_dim     -> int : Dimension of the embedding
                num_heads     -> int : Number of heads
                dropout       -> float : Dropout rate
        """
        super(Attention, self).__init__()
        self.dim = embed_dim
        self.mha = nn.MultiheadAttention(embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True, bias=True)

    def forward(self, x):
        """ Forward pass through the Attention module.
            :param:
                x -> torch.Tensor: Input tensor
            :return
                attention_value  -> torch.Tensor: Output the value of the attention
                attention_weight -> torch.Tensor: Output the weight of the attention
        """
        attention_value, attention_weight = self.mha(x, x, x)
        return attention_value, attention_weight


class TransformerEncoder(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout=0.):
        """ Initialize the Attention module.
            :param:
                dim       -> int : number of hidden dimension of attention
                depth     -> int : number of layer for the transformer
                heads     -> int : Number of heads
                mlp_dim   -> int : number of hidden dimension for mlp
                dropout   -> float : Dropout rate
        """
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        """ Forward pass through the Attention module.
            :param:
                x -> torch.Tensor: Input tensor
            :return
                x -> torch.Tensor: Output of the Transformer
                l_attn -> list(torch.Tensor): list of the attention
        """
        l_attn = []
        for attn, ff in self.layers:
            attention_value, attention_weight = attn(x)
            x = attention_value + x
            x = ff(x) + x
            l_attn.append(attention_weight)
        return x, l_attn
    


class MaskTransformerImg(nn.Module):
    def __init__(self, img_size=256, hidden_dim=768, codebook_size_img=1024,
                 num_tokens_img = 1024, depth=24, heads=8, mlp_dim=3072, dropout=0.1, nclass=1000):
        """ Initialize the Transformer model.
            :param:
                img_size       -> int:     Input image size (default: 256)
                hidden_dim     -> int:     Hidden dimension for the transformer (default: 768)
                codebook_size  -> int:     Size of the codebook (default: 1024)
                depth          -> int:     Depth of the transformer (default: 24)
                heads          -> int:     Number of attention heads (default: 8)
                mlp_dim        -> int:     MLP dimension (default: 3072)
                dropout        -> float:   Dropout rate (default: 0.1)
                nclass         -> int:     Number of classes (default: 1000)
        """
        super().__init__()
        self.nclass = nclass
        self.patch_size = img_size // 16
        self.codebook_size = codebook_size_img
        self.num_tokens_img = num_tokens_img
        self.tok_emb = nn.Embedding(self.codebook_size + 1, hidden_dim)  # +1 for the mask of the viz token, +1 for mask of the class
        self.pos_emb_img = nn.init.trunc_normal_(nn.Parameter(torch.zeros(1, self.num_tokens_img, hidden_dim)), 0., 0.02)

        # First layer before the Transformer block
        self.first_layer = nn.Sequential(
            nn.LayerNorm(hidden_dim, eps=1e-12),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim, eps=1e-12),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
        )

        self.transformer = TransformerEncoder(dim=hidden_dim, depth=depth, heads=heads, mlp_dim=mlp_dim, dropout=dropout)

        # Last layer after the Transformer block
        self.last_layer = nn.Sequential(
            nn.LayerNorm(hidden_dim, eps=1e-12),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim, eps=1e-12),
        )


    def forward(self, token, return_attn=False):
        """ Forward.
            :param:
                img_token      -> torch.LongTensor: bsize x 16 x 16, the encoded image tokens
                y              -> torch.LongTensor: condition class to generate
                drop_label     -> torch.BoolTensor: either or not to drop the condition
                return_attn    -> Bool: return the attn for visualization
            :return:
                logit:         -> torch.FloatTensor: bsize x path_size*path_size * 1024, the predicted logit
                attn:          -> list(torch.FloatTensor): list of attention for visualization
        """
        #b, w, h = token.size()
        tok_embeddings = self.tok_emb(token)

        # Position embedding
        tok_embeddings += self.pos_emb_img
        x = tok_embeddings

        # transformer forward pass
        x = self.first_layer(x)
        x, attn = self.transformer(x)
        x = self.last_layer(x)

        logit = torch.matmul(x, self.tok_emb.weight.T[:,:-1])

        if return_attn:  # return list of attention
            return logit, None, attn

        return logit, None




class MaskTransformer(nn.Module):
    def __init__(self, img_size=256, hidden_dim=768, codebook_size_img=1024, codebook_size_lidar=1024,
                 num_tokens_img = 1024, num_tokens_lidar = 4096, depth=24, heads=8, mlp_dim=3072, dropout=0.1, nclass=1000):
        """ Initialize the Transformer model.
            :param:
                img_size       -> int:     Input image size (default: 256)
                hidden_dim     -> int:     Hidden dimension for the transformer (default: 768)
                codebook_size  -> int:     Size of the codebook (default: 1024)
                depth          -> int:     Depth of the transformer (default: 24)
                heads          -> int:     Number of attention heads (default: 8)
                mlp_dim        -> int:     MLP dimension (default: 3072)
                dropout        -> float:   Dropout rate (default: 0.1)
                nclass         -> int:     Number of classes (default: 1000)
        """

        super().__init__()
        self.nclass = nclass
        self.patch_size = img_size // 16
        self.num_tokens_img = num_tokens_img
        self.num_tokens_lidar = num_tokens_lidar
        self.codebook_size_img = codebook_size_img
        self.codebook_size_lidar = codebook_size_lidar
        self.tok_emb_img = nn.Embedding(self.codebook_size_img + 1, hidden_dim)
        self.tok_emb_lidar = nn.Embedding(self.codebook_size_lidar + 1, hidden_dim)
        self.pos_emb_img = nn.init.trunc_normal_(nn.Parameter(torch.zeros(1, self.num_tokens_img, hidden_dim)), 0., 0.02)
        self.pos_emb_lidar = nn.init.trunc_normal_(nn.Parameter(torch.zeros(1, self.num_tokens_lidar, hidden_dim)), 0., 0.02)

        # First layer before the Transformer block
        self.first_layer = nn.Sequential(
            nn.LayerNorm(hidden_dim, eps=1e-12),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim, eps=1e-12),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
        )

        self.transformer = TransformerEncoder(dim=hidden_dim, depth=depth, heads=heads, mlp_dim=mlp_dim, dropout=dropout)

        # Last layer after the Transformer block
        self.last_layer = nn.Sequential(
            nn.LayerNorm(hidden_dim, eps=1e-12),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim, eps=1e-12),
        )

    def forward(self, token, return_attn=False):
        """ Forward.
            :param:
                img_token      -> torch.LongTensor: bsize x 16 x 16, the encoded image tokens
                y              -> torch.LongTensor: condition class to generate
                drop_label     -> torch.BoolTensor: either or not to drop the condition
                return_attn    -> Bool: return the attn for visualization
            :return:
                logit:         -> torch.FloatTensor: bsize x path_size*path_size * 1024, the predicted logit
                attn:          -> list(torch.FloatTensor): list of attention for visualization
        """
        #b, w, h = token.size()
        
        tok_img  = token[:,:self.num_tokens_img]
        tok_lidar= token[:,self.num_tokens_img:]

        """ print(tok_img.min())
        print(tok_img.max())
        print(self.codebook_size_img)
        print(tok_lidar.min())
        print(tok_lidar.max())
        print(self.codebook_size_lidar) """

        tok_embeddings_img = self.tok_emb_img(tok_img)
        tok_embeddings_lidar = self.tok_emb_lidar(tok_lidar)

        # Position embedding
        tok_embeddings_img += self.pos_emb_img
        tok_embeddings_lidar += self.pos_emb_lidar
        #print(tok_embeddings_img.shape)
        #print(tok_embeddings_lidar.shape)
        x = torch.cat((tok_embeddings_img,tok_embeddings_lidar), dim=1)
        x = x.contiguous()
        #print(f"x {x.shape}")
        #print(x.device)
        #print(x)

        # transformer forward pass
        x = self.first_layer(x)
        #print(f"x {x.shape}")
        x, attn = self.transformer(x)
        #print(f"x {x.shape}")
        x = self.last_layer(x)

        #print(f"x {x.shape}")
        tok_embeddings_img  = x[:,:self.num_tokens_img, :]
        tok_embeddings_lidar= x[:,self.num_tokens_img:, :]
        #print(f"tok_embeddings_img {tok_embeddings_img.shape}")
        #print(f"tok_embeddings_lidar {tok_embeddings_lidar.shape}")

        #print((self.tok_emb_img.weight.T).shape)
        #print((self.tok_emb_lidar.weight.T).shape)
        
        logit_img = torch.matmul(tok_embeddings_img, self.tok_emb_img.weight.T[:,:-1])  # :-1 to skip the mask token
        logit_lidar = torch.matmul(tok_embeddings_lidar, self.tok_emb_lidar.weight.T[:,:-1])
        #print(f"logit_img {logit_img.shape}")
        #print(f"logit_lidar {logit_lidar.shape}")

        #print(logit_lidar)

        #logit = torch.cat((logit_img, logit_lidar), dim=1)
        #print(f"logit {logit.shape}")

        if return_attn:  # return list of attention
            return logit_img, logit_lidar, attn

        return logit_img, logit_lidar
