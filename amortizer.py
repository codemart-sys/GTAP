import torch
import torch.nn as nn

class ShapleyAmortizer(nn.Module):

    def __init__(self, c, point_dim, image_dim, text_dim, n_coalitions=8, hidden=64, landmarks=24, heads=4, value_init=-0.69, value_scale=0.03, **_):
        super().__init__()
        if hidden % heads:
            raise ValueError('amortiser hidden width must be divisible by attention heads')
        self.landmarks = int(landmarks)
        self.n_coalitions = int(n_coalitions)
        self.landmark_proj = nn.Sequential(nn.Linear(c * 3, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.point_global = nn.Sequential(nn.Linear(point_dim, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.image_global = nn.Sequential(nn.Linear(image_dim, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.text_global = nn.Sequential(nn.Linear(text_dim, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.coalition_queries = nn.Parameter(torch.empty(n_coalitions, hidden))
        nn.init.normal_(self.coalition_queries, std=0.02)
        self.cross_attn = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.attn_norm = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(nn.Linear(hidden, hidden * 2), nn.GELU(), nn.Linear(hidden * 2, hidden))
        self.ffn_norm = nn.LayerNorm(hidden)
        self.value = nn.Linear(hidden, 1)
        with torch.no_grad():
            self.value.weight.mul_(value_scale)
            self.value.bias.fill_(value_init)

    @staticmethod
    def _landmark_indices(n, count, device):
        if count >= n:
            return torch.arange(n, device=device)
        return torch.linspace(0, n - 1, count, device=device).round().long()

    def forward(self, decoded, global_feature, image_global, text, image_effect):
        (b, _, n) = decoded.shape
        idx = self._landmark_indices(n, self.landmarks, decoded.device)
        point_tokens = decoded[:, :, idx]
        image_tokens = image_effect[:, :, idx]
        text_tokens = text.unsqueeze(-1).expand(-1, -1, idx.numel())
        local = torch.cat((point_tokens, image_tokens, text_tokens), dim=1).transpose(1, 2)
        local = self.landmark_proj(local)
        globals_ = torch.stack((self.point_global(global_feature), self.image_global(image_global), self.text_global(text)), dim=1)
        memory = torch.cat((globals_, local), dim=1)
        query = self.coalition_queries.unsqueeze(0).expand(b, -1, -1)
        (attended, _) = self.cross_attn(query, memory, memory, need_weights=False)
        query = self.attn_norm(query + attended)
        query = self.ffn_norm(query + self.ffn(query))
        return self.value(query).squeeze(-1)
