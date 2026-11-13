import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import Res50Encoder, Res101Encoder


class PrototypeDecoder(nn.Module):
    def __init__(self, in_channels=512 + 2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 256, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)
        self.conv_out = nn.Conv2d(256, 2, kernel_size=1, bias=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)) + x)
        return self.conv_out(x)


class SpatialOT(nn.Module):
    def __init__(self, num_components, alpha=0.8, pos_scale=1.0, assign_temp=1.0,
                 use_learned_temp=False):
        super().__init__()
        alpha = float(min(max(alpha, 1e-4), 1.0 - 1e-4))
        self.num_components = int(num_components)
        self.use_learned_temp = bool(use_learned_temp)
        self.alpha_logits = nn.Parameter(torch.full((self.num_components,), math.log(alpha / (1.0 - alpha))))
        self.log_scale = nn.Parameter(torch.full((self.num_components, 2), math.log(float(pos_scale))))
        self.log_assign_temp = nn.Parameter(torch.tensor(math.log(float(assign_temp))))

    def forward(self, feats, comps, coords):
        feats = F.normalize(feats, dim=1, eps=1e-6)
        comps = F.normalize(comps, dim=1, eps=1e-6)

        sim = feats @ comps.t()

        temp = torch.exp(self.log_assign_temp).clamp(0.05, 10.0) if self.use_learned_temp else 1.0
        q0 = F.softmax(sim / temp, dim=1)

        denom = q0.sum(dim=0, keepdim=True).t().clamp_min(1e-6)
        centers = (q0.t() @ coords) / denom

        scale = torch.exp(self.log_scale).clamp(1e-3, 100.0)
        d2 = ((coords.unsqueeze(1) - centers.unsqueeze(0)).pow(2) / scale.unsqueeze(0)).sum(dim=2)

        alpha = torch.sigmoid(self.alpha_logits).unsqueeze(0)
        return alpha * sim - (1.0 - alpha) * d2


class RandomWalkRefiner(nn.Module):
    def __init__(self, feat_dim=512, iters=5, guide_dim=16, use_learned_beta=False):
        super().__init__()
        self.use_learned_beta = bool(use_learned_beta)
        self.iters = int(iters)
        self.proj = nn.Conv2d(feat_dim, guide_dim, kernel_size=1, bias=False)
        self.log_beta = nn.Parameter(torch.tensor(-0.5))
        self.log_mu = nn.Parameter(torch.tensor(-1.0))

    def _shift(self, x, dy, dx):
        B, C, H, W = x.shape
        pad_y0 = max(dy, 0)
        pad_y1 = max(-dy, 0)
        pad_x0 = max(dx, 0)
        pad_x1 = max(-dx, 0)
        xpad = F.pad(x, (pad_x0, pad_x1, pad_y0, pad_y1), mode="replicate")
        y0 = pad_y0 - dy
        x0 = pad_x0 - dx
        return xpad[:, :, y0:y0 + H, x0:x0 + W]

    def forward(self, logits, qry_fts):
        q = torch.softmax(logits, dim=1)
        g = self.proj(qry_fts)
        g = F.normalize(g, dim=1, eps=1e-6)

        gU = self._shift(g, -1, 0)
        gD = self._shift(g, 1, 0)
        gL = self._shift(g, 0, -1)
        gR = self._shift(g, 0, 1)

        beta = torch.exp(self.log_beta).clamp(1e-3, 100.0) if self.use_learned_beta else 1.0
        mu = torch.sigmoid(self.log_mu)

        def w(ga, gb):
            d2 = (ga - gb).pow(2).sum(dim=1, keepdim=True)
            return torch.exp(-beta * d2.clamp(0.0, 10.0))

        wU, wD, wL, wR = w(g, gU), w(g, gD), w(g, gL), w(g, gR)

        for _ in range(self.iters):
            qU = self._shift(q, -1, 0)
            qD = self._shift(q, 1, 0)
            qL = self._shift(q, 0, -1)
            qR = self._shift(q, 0, 1)

            smooth = (wU * qU + wD * qD + wL * qL + wR * qR) / (wU + wD + wL + wR + 1e-6)
            q = (1.0 - mu) * q + mu * smooth
            q = q / (q.sum(dim=1, keepdim=True) + 1e-6)

        return torch.log(q.clamp_min(1e-6))


class UncertaintyAwareQueryPrototypeRectifier(nn.Module):
    """
    Parameter-light UQPR.

    This module does not use hand-crafted confidence thresholds. It softly rectifies
    support-derived prototypes with query evidence weighted by prediction reliability.
    The only extra learnable variables are temperature and fusion gates.
    """

    def __init__(self, safe_eps=1e-6, use_learned_temp=False):
        super().__init__()
        self.use_learned_temp = bool(use_learned_temp)
        self.safe_eps = float(safe_eps)
        self.log_assign_temp = nn.Parameter(torch.tensor(0.0))
        self.fg_gate_logit = nn.Parameter(torch.tensor(-3.0))
        self.bg_gate_logit = nn.Parameter(torch.tensor(-3.0))

    def _rectify_class(self, qry_fts, cls_prob, reliability, prototypes, gate_logit):
        """
        qry_fts:    [1, C, H, W]
        cls_prob:   [1, H, W]
        reliability:[1, H, W]
        prototypes: [K, C]
        """
        _, C, _, _ = qry_fts.shape
        K = prototypes.size(0)

        X = qry_fts.permute(0, 2, 3, 1).reshape(-1, C)
        P = prototypes
        Xn = F.normalize(X, dim=1, eps=self.safe_eps)
        Pn = F.normalize(P, dim=1, eps=self.safe_eps)

        # Soft query evidence. No hard threshold is used.
        weights = (cls_prob * reliability).reshape(-1, 1).detach()
        total_mass = weights.sum().clamp_min(self.safe_eps)

        temp = torch.exp(self.log_assign_temp).clamp(0.05, 10.0) if self.use_learned_temp else 1.0
        sim = (Xn @ Pn.t()) / temp
        assign = F.softmax(sim, dim=1) * weights

        comp_mass = assign.sum(dim=0).clamp_min(self.safe_eps)  # [K]
        Pq = (assign.t() @ X) / comp_mass.view(K, 1)

        # Evidence-adaptive gate. Components receiving more reliable query evidence
        # are rectified more strongly. The base gate is learnable and initialized small.
        evidence = (comp_mass / total_mass).detach()
        evidence = evidence / evidence.max().clamp_min(self.safe_eps)
        gate = torch.sigmoid(gate_logit + evidence).view(K, 1)

        refined = (1.0 - gate) * P + gate * Pq
        return refined

    def forward(self, qry_fts, logits, fg_prototypes, bg_prototypes):
        # Use detached initial prediction only for reliability estimation, avoiding
        # degenerate feedback through pseudo labels.
        with torch.no_grad():
            prob = torch.softmax(logits.detach(), dim=1)
            entropy = -(prob * torch.log(prob.clamp_min(self.safe_eps))).sum(dim=1)
            entropy = entropy / math.log(float(prob.size(1)))
            reliability = (1.0 - entropy).clamp(0.0, 1.0)

        fg_refined = self._rectify_class(
            qry_fts=qry_fts,
            cls_prob=prob[:, 1],
            reliability=reliability,
            prototypes=fg_prototypes,
            gate_logit=self.fg_gate_logit,
        )
        bg_refined = self._rectify_class(
            qry_fts=qry_fts,
            cls_prob=prob[:, 0],
            reliability=reliability,
            prototypes=bg_prototypes,
            gate_logit=self.bg_gate_logit,
        )
        return fg_refined, bg_refined


class FewShotSeg(nn.Module):
    def __init__(self, pretrained_weights=None, use_mb_head=False, sinkhorn_iters=7,
                 sinkhorn_eps=0.15, rw_iters=6, legacy_extra_scales=False,
                 legacy_uot_row_normalize=False):
        super().__init__()
        self.encoder = Res101Encoder(
            replace_stride_with_dilation=[True, True, False],
            pretrained_weights=pretrained_weights
        )

        self.scaler = 20.0
        self.legacy_uot_row_normalize = bool(legacy_uot_row_normalize)

        self.num_fg_prototypes = 2
        self.num_bg_prototypes = 4

        self.fg_components = nn.Parameter(torch.randn(self.num_fg_prototypes, 512))
        self.bg_components = nn.Parameter(torch.randn(self.num_bg_prototypes, 512))
        self.bg_prior_logits = nn.Parameter(torch.zeros(self.num_bg_prototypes))

        self.decoder = PrototypeDecoder(in_channels=512 + 2)

        self.use_sinkhorn_fg = True
        self.use_uot_bg = True
        self.sinkhorn_iters = int(sinkhorn_iters)
        self.sinkhorn_eps = float(sinkhorn_eps)

        self.min_pixels_per_proto = 30
        self.safe_eps = 1e-6

        self.fg_spatial_ot = SpatialOT(self.num_fg_prototypes, alpha=0.8, pos_scale=0.5,
                                       assign_temp=1.0, use_learned_temp=legacy_extra_scales)
        self.bg_spatial_ot = SpatialOT(self.num_bg_prototypes, alpha=0.6, pos_scale=1.0,
                                       assign_temp=1.0, use_learned_temp=legacy_extra_scales)

        self.rw_refine = RandomWalkRefiner(feat_dim=512, iters=int(rw_iters), guide_dim=16,
                                            use_learned_beta=legacy_extra_scales)
        self.rw_gate = nn.Parameter(torch.tensor(-1.5))

        self.use_rr_head = True
        self.rr_lambda = 1e-2
        self.rr_num_fg = 2048
        self.rr_num_bg = 2048
        self.rr_use_l2norm = True
        self.rr_detach_support = False
        self.rr_gate = nn.Parameter(torch.tensor(-2.0))

        self.use_mb_head = bool(use_mb_head)
        self.mb_eps = 1e-5
        self.mb_shrink = 0.2
        self.mb_min_pixels = 30
        self.mb_gate = nn.Parameter(torch.tensor(-2.0))


        # UQPR uses soft reliability weighting, without manual confidence thresholds.
        self.use_uqpr = True
        self.uqpr = UncertaintyAwareQueryPrototypeRectifier(safe_eps=self.safe_eps,
                                                         use_learned_temp=legacy_extra_scales)

    def _use_ot_now(self, enabled: bool, n_valid: int, K: int):
        if not enabled:
            return False
        if n_valid < K * self.min_pixels_per_proto:
            return False
        return True

    def sinkhorn_balanced(self, logits, n_iters=5, eps=0.15):
        with torch.amp.autocast('cuda', enabled=False):
            logits = logits.float()
            z = logits / float(eps)
            z = z - z.max(dim=1, keepdim=True).values
            Q = torch.exp(z).t().clamp_min(1e-12)
            Q = Q / Q.sum().clamp_min(1e-12)

            K, N = Q.shape
            r = torch.ones(K, device=Q.device, dtype=Q.dtype) / K
            c = torch.ones(N, device=Q.device, dtype=Q.dtype) / N

            for _ in range(n_iters):
                Q = Q * (r / Q.sum(dim=1).clamp_min(1e-12)).unsqueeze(1)
                Q = Q * (c / Q.sum(dim=0).clamp_min(1e-12)).unsqueeze(0)

            Q = Q / Q.sum(dim=0, keepdim=True).clamp_min(1e-12)
            return Q.t()

    def sinkhorn_unbalanced(self, logits, a=None, b=None, n_iters=5, eps=0.15, tau_a=2.0, tau_b=0.3):
        with torch.amp.autocast('cuda', enabled=False):
            logits = logits.float()
            N, K = logits.shape

            z = logits / float(eps)
            if self.legacy_uot_row_normalize:
                z = z - z.max(dim=1, keepdim=True).values
            Kmat = torch.exp(z.clamp(-60.0, 30.0)).clamp_min(1e-12)

            if a is None:
                a = torch.ones(N, device=Kmat.device, dtype=Kmat.dtype) / float(N)
            else:
                a = a.to(device=Kmat.device, dtype=Kmat.dtype)
                a = a / a.sum().clamp_min(1e-12)

            if b is None:
                b = torch.ones(K, device=Kmat.device, dtype=Kmat.dtype) / float(K)
            else:
                b = b.to(device=Kmat.device, dtype=Kmat.dtype)
                b = b / b.sum().clamp_min(1e-12)

            alpha = float(tau_a) / (float(tau_a) + float(eps))
            beta = float(tau_b) / (float(tau_b) + float(eps))

            u = torch.ones(N, device=Kmat.device, dtype=Kmat.dtype)
            v = torch.ones(K, device=Kmat.device, dtype=Kmat.dtype)

            for _ in range(n_iters):
                Kv = (Kmat @ v).clamp_min(1e-12)
                u = torch.pow(a / Kv, alpha)

                KTu = (Kmat.t() @ u).clamp_min(1e-12)
                v = torch.pow(b / KTu, beta)

            Q = (u.unsqueeze(1) * Kmat) * v.unsqueeze(0)
            if self.legacy_uot_row_normalize:
                return Q / Q.sum(dim=1, keepdim=True).clamp_min(1e-12)
            return Q

    def _sample_indices(self, idx_all: torch.Tensor, num: int):
        n = int(idx_all.numel())
        if n <= num:
            return idx_all
        perm = torch.randperm(n, device=idx_all.device)[:num]
        return idx_all[perm]

    def _solve_ridge_binary(self, X: torch.Tensor, y: torch.Tensor, lam: float):
        N, C = X.shape
        ones = torch.ones((N, 1), device=X.device, dtype=X.dtype)
        Xa = torch.cat([X, ones], dim=1)

        Y = torch.zeros((N, 2), device=X.device, dtype=X.dtype)
        Y.scatter_(1, y.view(-1, 1), 1.0)

        with torch.amp.autocast('cuda', enabled=False):
            Xa_f = Xa.float()
            Y_f = Y.float()
            Ct = Xa_f.shape[1]
            XtX = Xa_f.t() @ Xa_f
            reg = lam * torch.eye(Ct, device=Xa_f.device, dtype=Xa_f.dtype)
            A = XtX + reg
            B = Xa_f.t() @ Y_f
            try:
                Wab = torch.linalg.solve(A, B)
            except Exception:
                Wab = torch.linalg.pinv(A) @ B

        W = Wab[:-1, :].to(dtype=X.dtype)
        b = Wab[-1, :].to(dtype=X.dtype).view(2,)
        return W, b

    def _episode_rr_logits(self, supp_fts, supp_masks, qry_fts):
        device = supp_fts.device
        Sh, C, Hf, Wf = supp_fts.shape
        H, W = supp_masks.shape[-2:]

        fg_masks = supp_masks.view(Sh, 1, H, W).float()
        fg_masks_f = F.interpolate(fg_masks, size=(Hf, Wf), mode="nearest")
        bg_masks_f = 1.0 - fg_masks_f

        Xs = supp_fts.permute(0, 2, 3, 1).reshape(-1, C)
        fg_flat = fg_masks_f.view(-1) > 0.5
        bg_flat = bg_masks_f.view(-1) > 0.5

        if not fg_flat.any():
            fg_flat = torch.ones_like(fg_flat, dtype=torch.bool)
        if not bg_flat.any():
            bg_flat = torch.ones_like(bg_flat, dtype=torch.bool)

        fg_idx = torch.nonzero(fg_flat, as_tuple=False).view(-1)
        bg_idx = torch.nonzero(bg_flat, as_tuple=False).view(-1)

        fg_idx = self._sample_indices(fg_idx, int(self.rr_num_fg))
        bg_idx = self._sample_indices(bg_idx, int(self.rr_num_bg))

        X_fg = Xs[fg_idx]
        X_bg = Xs[bg_idx]

        if self.rr_detach_support:
            X_fg = X_fg.detach()
            X_bg = X_bg.detach()

        X = torch.cat([X_bg, X_fg], dim=0)
        y = torch.cat([
            torch.zeros((X_bg.shape[0],), device=device, dtype=torch.long),
            torch.ones((X_fg.shape[0],), device=device, dtype=torch.long)
        ], dim=0)

        if self.rr_use_l2norm:
            X = F.normalize(X, dim=1, eps=self.safe_eps)

        Wrr, brr = self._solve_ridge_binary(X, y, lam=float(self.rr_lambda))

        Q = qry_fts
        if self.rr_use_l2norm:
            Q = F.normalize(Q, dim=1, eps=self.safe_eps)

        Qf = Q.permute(0, 2, 3, 1).reshape(-1, C)
        logits = Qf @ Wrr + brr.view(1, 2)
        logits = logits.view(1, Hf, Wf, 2).permute(0, 3, 1, 2).contiguous()
        return logits

    def _episode_mb_logits(self, supp_fts, supp_masks, qry_fts):
        Sh, C, Hf, Wf = supp_fts.shape
        H, W = supp_masks.shape[-2:]

        m = supp_masks.view(Sh, 1, H, W).float()
        m_f = F.interpolate(m, size=(Hf, Wf), mode="nearest")

        Xs = supp_fts.permute(0, 2, 3, 1).reshape(-1, C)
        fg_flat = (m_f.view(-1) > 0.5)
        bg_flat = ~fg_flat

        if fg_flat.sum() < self.mb_min_pixels:
            fg_flat = torch.ones_like(fg_flat, dtype=torch.bool)
            bg_flat = ~fg_flat
        if bg_flat.sum() < self.mb_min_pixels:
            bg_flat = torch.ones_like(bg_flat, dtype=torch.bool)

        X_fg = Xs[torch.nonzero(fg_flat, as_tuple=False).view(-1)]
        X_bg = Xs[torch.nonzero(bg_flat, as_tuple=False).view(-1)]

        mu_fg = X_fg.mean(dim=0, keepdim=True)
        mu_bg = X_bg.mean(dim=0, keepdim=True)

        var_fg = (X_fg - mu_fg).pow(2).mean(dim=0, keepdim=True)
        var_bg = (X_bg - mu_bg).pow(2).mean(dim=0, keepdim=True)

        if self.mb_shrink > 0:
            var_fg = (1.0 - float(self.mb_shrink)) * var_fg + float(self.mb_shrink) * torch.ones_like(var_fg)
            var_bg = (1.0 - float(self.mb_shrink)) * var_bg + float(self.mb_shrink) * torch.ones_like(var_bg)

        var_fg = var_fg.clamp_min(float(self.mb_eps))
        var_bg = var_bg.clamp_min(float(self.mb_eps))

        Q = qry_fts.permute(0, 2, 3, 1).reshape(-1, C)

        d_fg = (Q - mu_fg).pow(2) / var_fg + torch.log(var_fg)
        d_bg = (Q - mu_bg).pow(2) / var_bg + torch.log(var_bg)

        s_fg = -0.5 * d_fg.sum(dim=1, keepdim=True)
        s_bg = -0.5 * d_bg.sum(dim=1, keepdim=True)

        logits = torch.cat([s_bg, s_fg], dim=1)
        logits = logits.view(1, Hf, Wf, 2).permute(0, 3, 1, 2).contiguous()
        return logits

    def _proto_conf_weights(self, supp_fts, supp_masks, prototypes, is_fg: bool):
        Sh, C, Hf, Wf = supp_fts.shape
        H, W = supp_masks.shape[-2:]

        m = supp_masks.view(Sh, 1, H, W).float()
        m_f = F.interpolate(m, size=(Hf, Wf), mode="nearest")
        if is_fg:
            valid = (m_f.view(-1) > 0.5)
        else:
            valid = (m_f.view(-1) <= 0.5)

        Xs = supp_fts.permute(0, 2, 3, 1).reshape(-1, C)
        if valid.sum() < 10:
            valid = torch.ones_like(valid, dtype=torch.bool)
        X = Xs[valid]

        X = F.normalize(X, dim=1, eps=self.safe_eps)
        P = F.normalize(prototypes, dim=1, eps=self.safe_eps)

        sim = X @ P.t()
        s = sim.mean(dim=0)
        w = torch.softmax(s, dim=0).clamp_min(1e-6)
        return w


    def _decode_with_prototypes(self, cur_qry_fts, cur_supp_fts, cur_supp_masks, fg_prototypes, bg_prototypes):
        _, C, _, _ = cur_qry_fts.shape

        w_fg = self._proto_conf_weights(cur_supp_fts, cur_supp_masks, fg_prototypes, is_fg=True)
        w_bg = self._proto_conf_weights(cur_supp_fts, cur_supp_masks, bg_prototypes, is_fg=False)

        qry_norm = F.normalize(cur_qry_fts, dim=1, eps=self.safe_eps)
        fg_proto_norm = F.normalize(fg_prototypes, p=2, dim=1, eps=self.safe_eps)
        bg_proto_norm = F.normalize(bg_prototypes, p=2, dim=1, eps=self.safe_eps)

        Kf = fg_proto_norm.size(0)
        Kb = bg_proto_norm.size(0)

        resp_fg_all = self.scaler * F.conv2d(qry_norm, fg_proto_norm.view(Kf, C, 1, 1))
        resp_fg = torch.logsumexp(resp_fg_all + torch.log(w_fg.view(1, Kf, 1, 1)), dim=1, keepdim=True)

        resp_bg_all = self.scaler * F.conv2d(qry_norm, bg_proto_norm.view(Kb, C, 1, 1))
        resp_bg = torch.logsumexp(resp_bg_all + torch.log(w_bg.view(1, Kb, 1, 1)), dim=1, keepdim=True)

        dec_in = torch.cat([cur_qry_fts, resp_fg, resp_bg], dim=1)
        logits_q = self.decoder(dec_in)
        return logits_q

    def forward(self, supp_imgs, supp_mask, qry_imgs, train=False):
        self.n_ways = len(supp_imgs)
        self.n_shots = len(supp_imgs[0])
        self.n_queries = len(qry_imgs)
        assert self.n_ways == 1
        assert self.n_queries == 1

        qry_bs = qry_imgs[0].shape[0]
        supp_bs = supp_imgs[0][0].shape[0]
        assert qry_bs == supp_bs, f"qry_bs({qry_bs}) != supp_bs({supp_bs})"

        img_size = supp_imgs[0][0].shape[-2:]

        supp_mask = torch.stack([torch.stack(way, dim=0) for way in supp_mask], dim=0)
        supp_mask = supp_mask.view(supp_bs, self.n_ways, self.n_shots, *img_size)

        imgs_concat = torch.cat(
            [torch.cat(way, dim=0) for way in supp_imgs] + [torch.cat(qry_imgs, dim=0)],
            dim=0
        )

        img_fts, _ = self.encoder(imgs_concat)

        supp_fts = img_fts[:self.n_ways * self.n_shots * supp_bs].view(
            supp_bs, self.n_ways, self.n_shots, -1, *img_fts.shape[-2:]
        )
        qry_fts = img_fts[self.n_ways * self.n_shots * supp_bs:].view(
            qry_bs, self.n_queries, -1, *img_fts.shape[-2:]
        )

        outputs = []
        prc_loss_list = []

        _, _, C, Hf, Wf = qry_fts.shape

        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, Hf, device=qry_fts.device, dtype=qry_fts.dtype),
            torch.linspace(-1.0, 1.0, Wf, device=qry_fts.device, dtype=qry_fts.dtype),
            indexing="ij"
        )
        coords = torch.stack([yy, xx], dim=-1).view(-1, 2)

        for epi in range(supp_bs):
            cur_supp_fts = supp_fts[epi, 0]
            cur_supp_masks = supp_mask[epi, 0]
            cur_qry_fts = qry_fts[epi, 0].unsqueeze(0)

            fg_prototypes = self.get_fg_mixture_prototypes(cur_supp_fts, cur_supp_masks, coords)
            bg_prototypes = self.get_bg_mixture_prototypes(cur_supp_fts, cur_supp_masks, coords)

            # Stage 1: initial prediction using support-derived prototypes.
            logits_q = self._decode_with_prototypes(
                cur_qry_fts, cur_supp_fts, cur_supp_masks, fg_prototypes, bg_prototypes
            )

            # Stage 2: uncertainty-aware soft query rectification. The support prototypes
            # used by PRC are kept unchanged; only prediction prototypes are rectified.
            if self.use_uqpr:
                fg_pred_prototypes, bg_pred_prototypes = self.uqpr(
                    cur_qry_fts, logits_q, fg_prototypes, bg_prototypes
                )
                logits_q = self._decode_with_prototypes(
                    cur_qry_fts, cur_supp_fts, cur_supp_masks, fg_pred_prototypes, bg_pred_prototypes
                )

            if self.use_rr_head:
                logits_rr = self._episode_rr_logits(cur_supp_fts, cur_supp_masks, cur_qry_fts)
                g_rr = torch.sigmoid(self.rr_gate)
                logits_q = logits_q + g_rr * logits_rr

            if self.use_mb_head:
                logits_mb = self._episode_mb_logits(cur_supp_fts, cur_supp_masks, cur_qry_fts)
                g_mb = torch.sigmoid(self.mb_gate)
                logits_q = logits_q + g_mb * logits_mb

            logits_rw = self.rw_refine(logits_q, cur_qry_fts)
            g = torch.sigmoid(self.rw_gate)
            logits_q = (1.0 - g) * logits_q + g * logits_rw

            logits_up = F.interpolate(logits_q, size=img_size, mode="bilinear", align_corners=True)

            if train:
                outputs.append(logits_up)
                prc_loss = self.compute_support_reconstruction_loss(
                    cur_supp_fts, cur_supp_masks, fg_prototypes, bg_prototypes
                )
                prc_loss_list.append(prc_loss)
            else:
                outputs.append(torch.softmax(logits_up, dim=1))

        output = torch.stack(outputs, dim=1).view(-1, *outputs[0].shape[1:])

        if train:
            device = output.device
            prc_loss_mean = torch.stack(prc_loss_list).mean() if prc_loss_list else torch.zeros(1, device=device)
            return output, prc_loss_mean
        else:
            return output

    def get_fg_mixture_prototypes(self, supp_fts, supp_masks, coords):
        device = supp_fts.device
        Sh, C, Hf, Wf = supp_fts.shape
        H, W = supp_masks.shape[-2:]

        fg_masks = supp_masks.view(Sh, 1, H, W).float()
        fg_masks_resized = F.interpolate(fg_masks, size=(Hf, Wf), mode="nearest")

        feats_flat = supp_fts.permute(0, 2, 3, 1).reshape(-1, C)
        fg_masks_flat = fg_masks_resized.view(-1)

        valid = fg_masks_flat > 0.5
        if valid.sum() == 0:
            valid = torch.ones_like(fg_masks_flat, dtype=torch.bool)

        fg_feats = feats_flat[valid]
        Nf = int(fg_feats.shape[0])
        if fg_feats.numel() == 0:
            fg_feats = feats_flat
            Nf = int(fg_feats.shape[0])

        coords_rep = coords.repeat(Sh, 1)
        coords_v = coords_rep[valid]

        logits = self.fg_spatial_ot(fg_feats, self.fg_components, coords_v)

        K = self.fg_components.size(0)
        if self._use_ot_now(enabled=self.use_sinkhorn_fg, n_valid=Nf, K=K):
            q = self.sinkhorn_balanced(logits, n_iters=self.sinkhorn_iters, eps=self.sinkhorn_eps)
        else:
            q = F.softmax(logits, dim=1)

        weighted_sum = torch.matmul(q.t(), fg_feats)
        weights_per_comp = q.sum(dim=0, keepdim=True).t()
        fg_prototypes = weighted_sum / (weights_per_comp + 1e-6)

        if torch.isnan(fg_prototypes).any() or torch.isinf(fg_prototypes).any():
            fg_prototypes = fg_feats.mean(dim=0, keepdim=True).repeat(K, 1)

        return fg_prototypes.to(device)

    def get_bg_mixture_prototypes(self, supp_fts, supp_masks, coords):
        device = supp_fts.device
        Sh, C, Hf, Wf = supp_fts.shape
        H, W = supp_masks.shape[-2:]

        fg_masks = supp_masks.view(Sh, 1, H, W).float()
        fg_masks_resized = F.interpolate(fg_masks, size=(Hf, Wf), mode="nearest")
        bg_masks = 1.0 - fg_masks_resized

        feats_flat = supp_fts.permute(0, 2, 3, 1).reshape(-1, C)
        bg_masks_flat = bg_masks.view(-1)

        valid = bg_masks_flat > 0.5
        if valid.sum() == 0:
            valid = torch.ones_like(bg_masks_flat, dtype=torch.bool)

        bg_feats = feats_flat[valid]
        Nb = int(bg_feats.shape[0])
        if bg_feats.numel() == 0:
            bg_feats = feats_flat
            Nb = int(bg_feats.shape[0])

        coords_rep = coords.repeat(Sh, 1)
        coords_v = coords_rep[valid]

        logits = self.bg_spatial_ot(bg_feats, self.bg_components, coords_v)

        K = self.bg_components.size(0)

        if self._use_ot_now(enabled=self.use_uot_bg, n_valid=Nb, K=K):
            b = torch.softmax(self.bg_prior_logits, dim=0)
            q = self.sinkhorn_unbalanced(
                logits,
                a=None,
                b=b,
                n_iters=self.sinkhorn_iters,
                eps=self.sinkhorn_eps
            )
        else:
            q = F.softmax(logits, dim=1)

        weighted_sum = torch.matmul(q.t(), bg_feats)
        weights_per_comp = q.sum(dim=0, keepdim=True).t()
        bg_prototypes = weighted_sum / (weights_per_comp + 1e-6)

        if torch.isnan(bg_prototypes).any() or torch.isinf(bg_prototypes).any():
            bg_prototypes = bg_feats.mean(dim=0, keepdim=True).repeat(K, 1)

        return bg_prototypes.to(device)

    def compute_support_reconstruction_loss(self, supp_fts, supp_masks, fg_prototypes, bg_prototypes):
        device = supp_fts.device
        Sh, C, Hf, Wf = supp_fts.shape
        H, W = supp_masks.shape[-2:]

        fg_masks = supp_masks.view(Sh, 1, H, W).float()
        fg_masks_resized = F.interpolate(fg_masks, size=(Hf, Wf), mode="nearest")
        labels = fg_masks_resized.view(-1).to(device)

        X = supp_fts.permute(0, 2, 3, 1).reshape(-1, C).to(device)
        X = F.normalize(X, p=2, dim=1, eps=self.safe_eps)

        def ridge_recon_residual(Xn, P, eps=1e-4):
            with torch.amp.autocast('cuda', enabled=False):
                Xn = Xn.float()
                P = F.normalize(P.float(), p=2, dim=1, eps=self.safe_eps)
                K = P.size(0)
                G = P @ P.t()
                G = G + eps * torch.eye(K, device=P.device, dtype=P.dtype)
                B = P @ Xn.t()
                try:
                    A = torch.linalg.solve(G, B)
                except Exception:
                    A = torch.linalg.pinv(G) @ B
                X_hat = A.t() @ P
                return (Xn - X_hat).pow(2).sum(dim=1)

        r_fg = ridge_recon_residual(X, fg_prototypes)
        r_bg = ridge_recon_residual(X, bg_prototypes)

        logits = (r_bg - r_fg)
        return F.binary_cross_entropy_with_logits(logits, labels, reduction="mean")