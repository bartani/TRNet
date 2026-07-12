import torch
import torch.nn.functional as F

def generate_greedy_max_distance_points(K, dim=512, pool_size=10000):
    # Generate a large pool of normalized random vectors
    pool = F.normalize(torch.randn(pool_size, dim), p=2, dim=1)

    # Select first point randomly (make sure it's 1D, not 2D)
    idx = torch.randint(0, pool_size, (1,)).item()
    selected = [pool[idx]]  # shape: [512]

    for _ in range(K - 1):
        selected_tensor = torch.stack(selected)  # shape: [num_selected, 512]
        # Compute distances from each point in the pool to the selected set
        dists = torch.cdist(pool, selected_tensor)  # shape: [pool_size, num_selected]
        min_dist = dists.min(dim=1).values  # Closest distance to any selected point
        next_idx = min_dist.argmax().item()  # Pick the farthest-from-closest
        selected.append(pool[next_idx])  # Add the new point

    return torch.stack(selected)

def save(points, K):
    torch.save(points, f"{K}_points.pt")

def load(k):
    loaded_points = torch.load(f"{K}_points.pt")
    return loaded_points

if __name__ == "__main__":
    # Example usage
    K = 4
    points = generate_greedy_max_distance_points(K)
    print(points.shape)  # torch.Size([10, 512])

    save(points, K)

    p0 = points[1]
    p1 = points[2]

    # Compute Euclidean distance
    distance = torch.norm(p0 - p1, p=2)
    print(f"Distance between points[0] and points[1]: {distance.item():.4f}")


    p = load(K)

    
    p0 = p[1]
    p1 = p[2]

    # Compute Euclidean distance
    distance = torch.norm(p0 - p1, p=2)
    print(f"Distance between points[0] and points[1]: {distance.item():.4f}")
