import torch
import torch.nn as nn
import torch.nn.functional as F


def square_distance(src, dst):
    """
    Calculate Euclidean distance between each two points.
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).reshape(B, N, 1)
    dist += torch.sum(dst ** 2, -1).reshape(B, 1, M)
    return dist


def farthest_point_sample(xyz, npoint):
    """
    Farthest Point Sampling
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].reshape(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Query ball point
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).reshape(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points):
    """
    Sample and group
    """
    B, N, C = xyz.shape
    S = npoint
    
    fps_idx = farthest_point_sample(xyz, npoint)
    new_xyz = torch.gather(xyz, 1, fps_idx.unsqueeze(-1).expand(-1, -1, C))
    
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = torch.gather(xyz, 1, idx.reshape(B, -1, 1).expand(-1, -1, C)).reshape(B, S, nsample, C)
    grouped_xyz_norm = grouped_xyz - new_xyz.reshape(B, S, 1, C)
    
    if points is not None:
        grouped_points = torch.gather(points, 1, idx.reshape(B, -1, 1).expand(-1, -1, points.size(-1))).reshape(B, S, nsample, -1)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
    else:
        new_points = grouped_xyz_norm
    
    return new_xyz, new_points


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all=False):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
    
    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)  # [B, N, C]
        if points is not None:
            points = points.permute(0, 2, 1)  # [B, N, D]
        
        # Global pooling (when npoint is None)
        if self.npoint is None:
            B, N, C = xyz.shape
            new_xyz = torch.zeros(B, 1, C).to(xyz.device)  # [B, 1, C]
            if points is not None:
                new_points = torch.cat([xyz, points], dim=-1)  # [B, N, C+D]
            else:
                new_points = xyz  # [B, N, C]
            new_points = new_points.unsqueeze(2)  # [B, N, 1, C+D]
            new_points = new_points.permute(0, 3, 2, 1)  # [B, C+D, 1, N]
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
            # new_points: [B, npoint, nsample, C+D]
            new_points = new_points.permute(0, 3, 2, 1)  # [B, C+D, nsample, npoint]
        
        # new_points: [B, C+D, nsample, npoint]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        
        # Max pooling
        new_points = torch.max(new_points, 2)[0]  # [B, D', npoint or N]
        new_xyz = new_xyz.permute(0, 2, 1)  # [B, C, npoint or 1]
        return new_xyz, new_points


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel
    
    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D1, N]
            points2: input points data, [B, D2, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)  # [B, N, C]
        xyz2 = xyz2.permute(0, 2, 1)  # [B, S, C]
        
        points2 = points2.permute(0, 2, 1)  # [B, S, D2]
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape
        
        if S == 1:
            # Global feature case - simply repeat for all points
            interpolated_points = points2.expand(B, N, -1)  # [B, N, D2]
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]
            
            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm  # [B, N, 3]
            
            # Gather the 3 nearest neighbors for each point
            idx_expanded = idx.unsqueeze(-1).expand(-1, -1, -1, points2.size(-1))  # [B, N, 3, D2]
            points2_expanded = points2.unsqueeze(1).expand(-1, N, -1, -1)  # [B, N, S, D2]
            gathered_points = torch.gather(points2_expanded, 2, idx_expanded)  # [B, N, 3, D2]
            
            # Weighted sum
            weight_expanded = weight.unsqueeze(-1)  # [B, N, 3, 1]
            interpolated_points = torch.sum(gathered_points * weight_expanded, dim=2)  # [B, N, D2]
        
        if points1 is not None:
            points1 = points1.permute(0, 2, 1)  # [B, N, D1]
            new_points = torch.cat([points1, interpolated_points], dim=-1)  # [B, N, D1+D2]
        else:
            new_points = interpolated_points  # [B, N, D2]
        
        new_points = new_points.permute(0, 2, 1)  # [B, D1+D2, N]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        
        return new_points


class PointNet2Segmentation(nn.Module):
    def __init__(self, num_classes):
        super(PointNet2Segmentation, self).__init__()
        
        # Set Abstraction layers
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=3, mlp=[64, 64, 128])
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256])
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024])
        
        # Feature Propagation layers
        self.fp3 = PointNetFeaturePropagation(in_channel=1024 + 256, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=256 + 128, mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel=128, mlp=[128, 128, 128])
        
        # Final classification layers
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)
    
    def forward(self, xyz):
        """
        Input:
            xyz: input points position data, [B, C, N]
        Return:
            x: segmentation scores for each point, [B, N, num_classes]
        """
        l0_xyz = xyz
        l0_points = None
        
        # Set Abstraction layers
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)  # l1_points: [B, 128, 512]
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)  # l2_points: [B, 256, 128]
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)  # l3_points: [B, 1024, 1]
        
        # Feature Propagation layers
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)  # [B, 256, 128]
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)  # [B, 128, 512]
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)  # [B, 128, N]
        
        # Final classification
        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        
        return x


if __name__ == "__main__":
    model = PointNet2Segmentation(num_classes=2)
    xyz = torch.randn(2, 3, 2048)
    output = model(xyz)
    print(f"Input shape: {xyz.shape}")
    print(f"Output shape: {output.shape}")