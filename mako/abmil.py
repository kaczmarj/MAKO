"""Attention-based multiple instance learning implementation.

Modified from https://github.com/AMLab-Amsterdam/AttentionDeepMIL.

References
----------
Ilse, M., Tomczak, J. M., & Welling, M. (2018). Attention-based Deep Multiple
Instance Learning. arXiv preprint arXiv:1802.04712.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F


# Adapted from
# https://github.com/AMLab-Amsterdam/AttentionDeepMIL/blob/bf1ee90af01eaec6e65fd8c685f996ac868934cf/model.py#L27-L31
def AttentionLayer(L: int, D: int, K: int):
    """Attention layer (without gating)."""
    return nn.Sequential(nn.Linear(L, D), nn.Tanh(), nn.Linear(D, K))  # NxK


# Adapted from
# https://github.com/AMLab-Amsterdam/AttentionDeepMIL/blob/bf1ee90af01eaec6e65fd8c685f996ac868934cf/model.py#L72C33-L128
class GatedAttentionLayer(nn.Module):
    """Gated attention layer."""

    def __init__(self, L: int, D: int, K: int, *, dropout: float = 0.25):
        super().__init__()
        self.L = L
        self.D = D
        self.K = K
        self.dropout = dropout

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Dropout(self.dropout),
        )
        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid(),
            nn.Dropout(self.dropout),
        )
        self.attention_weights = nn.Linear(self.D, self.K)

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        A_V = self.attention_V(H)  # NxD
        A_U = self.attention_U(H)  # NxD
        A = self.attention_weights(A_V * A_U)  # element wise multiplication # NxK
        return A


class AttentionMILModelOutput(NamedTuple):
    """A container for the outputs of an ABMIL model."""

    logits: torch.Tensor
    attention: torch.Tensor


class AttentionMILModel(nn.Module):
    """
    Attention-based multiple instance learning (ABMIL) model for weakly-supervised learning.

    This model processes instance-level feature representations using a fully connected
    encoder, applies an attention mechanism to compute instance importance, and aggregates
    instance-level features into a bag-level representation for classification or regression.

    Parameters
    ----------
    in_features : int
        Dimensionality of the input feature vectors.
    L : int
        Dimensionality of the encoded feature space.
    D : int
        Dimensionality of the hidden representation in the attention mechanism.
    K : int, optional
        Number of attention heads. Default is 1.
    num_classes : int
        Number of output classes for classification.
    dropout : float, optional
        Dropout probability applied in the encoder and attention layers. Default is 0.25.
    gated_attention : bool, optional
        Whether to use a gated attention mechanism. If False, a standard attention
        mechanism is used. Default is True.

    Notes
    -----
    The model first encodes patch-level features, applies attention-based feature weighting,
    aggregates instance representations, and maps the final bag representation to an output
    prediction.
    """

    def __init__(
        self,
        *,
        in_features: int,
        L: int,
        D: int,
        K: int = 1,
        num_classes: int,
        dropout: float = 0.25,
        gated_attention: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.L = L
        self.D = D
        self.K = K
        self.num_classes = num_classes
        self.dropout = dropout
        self.gated_attention = gated_attention

        self.encoder = nn.Sequential(
            nn.Linear(in_features, L), nn.ReLU(), nn.Dropout(dropout)
        )
        if gated_attention:
            self.attention_weights = GatedAttentionLayer(L=L, D=D, K=K, dropout=dropout)
        else:
            self.attention_weights = AttentionLayer(L=L, D=D, K=K)
        self.head = nn.Linear(L, self.num_classes)

    def forward(self, H: torch.Tensor) -> AttentionMILModelOutput:
        """
        Forward pass of the ABMIL model.

        Parameters
        ----------
        H : torch.Tensor
            Input tensor of shape (N, in_features), where N is the number of instances.

        Returns
        -------
        AttentionMILModelOutput
            Named tuple containing:
            - logits (torch.Tensor): Model predictions of shape (1, num_classes).
            - attention (torch.Tensor): Raw attention scores before softmax.
        """
        # H.shape is N x in_features
        assert H.ndim == 2, f"Expected H to have 2 dimensions but got {H.ndim}"

        H = self.encoder(H)  # NxL
        A = self.attention_weights(H)  # NxK
        A_raw = A
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N
        M = torch.mm(A, H)  # KxL
        logits = self.head(M)  # 1xK
        return AttentionMILModelOutput(logits=logits, attention=A_raw)
