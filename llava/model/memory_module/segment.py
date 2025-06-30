import torch

def cal_depth_score(sim_scores):
    n = sim_scores.shape[0]
    depth_scores = torch.zeros(sim_scores.size(), dtype=sim_scores.dtype, device=sim_scores.device)
    # clip = min(max(n//10, 2), 5) # adopt clip to improve efficiency
    for i in range(n):
        lpeak = sim_scores[i]
        for li in range(i-1, -1, -1):
            if sim_scores[li] >= lpeak:
                lpeak = sim_scores[li]
            #如果左侧的相似度 sim_scores[li] 大于等于当前的最大峰值 lpeak，更新 lpeak，否则停止，说明左侧的相似度已经开始下降
            else:
                break
        rpeak = sim_scores[i]
        for ri in range(i+1, n):
            if sim_scores[ri] >= rpeak:
                rpeak = sim_scores[ri]
            #如果右侧的相似度 sim_scores[ri] 大于等于当前的最大峰值 rpeak，更新 rpeak，否则停止，说明右侧的相似度已经开始下降
            else:
                break
        #当depth score的值越大，说明当前时间点相对于左右的相似度越小，即越可能是segment的边界
        depth_scores[i] = lpeak + rpeak - 2 * sim_scores[i]
    return depth_scores


def segment(features, alpha=0.5, k=None):
    if features.shape[0] == 1:
        return [0], torch.zeros(1)

    sim_scores = torch.cosine_similarity(features[:-1, :], features[1:, :])
    depth_scores = cal_depth_score(sim_scores)

    if k is not None:
        boundaries = torch.topk(depth_scores, k).indices.sort()[0]
    else:
        std, mean = torch.std_mean(depth_scores)
        thresh = mean + alpha * std
        condition = depth_scores > thresh
        boundaries = condition.nonzero().squeeze(-1)
        # if len(boundaries) > 15:
        #     boundaries = torch.topk(depth_scores, 15).indices.sort()[0]
    boundaries = boundaries.tolist()

    if type(boundaries) == int or boundaries == [] or boundaries[-1] != features.shape[0]-1:
        boundaries.append(features.shape[0])

    boundaries = sorted(set(boundaries))
    return boundaries, depth_scores


def adjusted_segment(features, alpha=0.5, k=None, min_distance=32, max_distance=64):
    """
    Segment a sequence of features into segments based on cosine similarity.

    Parameters:
      features: tensor of shape (t, d)
      alpha: parameter for thresholding depth scores
      k: if provided, select top-k boundaries based on depth scores
      min_distance: minimum allowed gap between consecutive boundaries
      max_distance: maximum allowed gap between consecutive boundaries;
                    if a gap is larger, extra boundaries will be inserted evenly.
    """
    # If there is only one time point, return [0] as the only boundary.
    if features.shape[0] == 1:
        return [0]

    # Compute cosine similarity between adjacent features
    sim_scores = torch.cosine_similarity(features[:-1, :], features[1:, :])
    depth_scores = cal_depth_score(sim_scores)

    # Determine candidate boundaries based on either top-k or thresholding
    if k is not None:
        boundaries = torch.topk(depth_scores, k).indices.sort()[0]
    else:
        std, mean = torch.std_mean(depth_scores)
        thresh = mean + alpha * std
        condition = depth_scores > thresh
        boundaries = condition.nonzero().squeeze(-1)
        if len(boundaries) > 15:
            boundaries = torch.topk(depth_scores, 15).indices.sort()[0]

    boundaries = boundaries.tolist()

    # Always include the last time point as a boundary
    if not boundaries or boundaries[-1] != features.shape[0]:
        # the right boundary is not selected by the segment, so the last boundary must be len
        boundaries.append(features.shape[0])
    if boundaries[0] != 0:
        boundaries.insert(0, 0)
    # Remove duplicates and sort
    boundaries = sorted(set(boundaries))
    print("boundaries before adjusted: ", boundaries)


    adjusted_boundaries = [boundaries[0]]
    for idx, b in enumerate(boundaries[1:-1], start=1):
        gap = b - adjusted_boundaries[-1]
        if gap < min_distance:
            # Skip if too close
            continue
        if gap > max_distance:
            # Compute number of extra boundaries to insert uniformly.
            X = int(gap / max_distance)
            start = adjusted_boundaries[-1]
            # Insert extra boundaries uniformly between start and b
            for i in range(1, X + 1):
                new_boundary = start + round(gap * i / (X + 1))
                # Ensure that the new boundary is strictly between start and b
                if new_boundary > adjusted_boundaries[-1] and new_boundary < b:
                    adjusted_boundaries.append(new_boundary)
        # Always add the candidate boundary.
        adjusted_boundaries.append(b)
    # Check if we should add the final boundary
    last_boundary = features.shape[0]
    gap = last_boundary - adjusted_boundaries[-1]

    if gap >= min_distance:
        adjusted_boundaries.append(last_boundary)
    elif adjusted_boundaries[-1] == 0:
        adjusted_boundaries.append(last_boundary)
    else:
        # If last segment is too small, consider merging it
        # by removing the previous boundary (if possible)
        adjusted_boundaries[-1] = last_boundary  # merge small segment into previous
    # print("boundaries after adjusted: ", adjusted_boundaries)

    return adjusted_boundaries


def uniform_segment(features, d=32):
    """
    Naively segment `features` into chunks of size d.
    - The first chunk can be smaller than d if T % d != 0.
    - The last chunk will always be exactly d in length unless T <= d (then there's just one chunk).

    :param features: A tensor or array of shape (T, D), where T is the temporal length.
    :param d: The fixed chunk size.
    :return: A list of boundary indices (the ends of each chunk).
    """
    T = features.shape[0]
    # If the entire sequence is shorter than or equal to d, just return one boundary
    if T <= d:
        return [0, T]

    # Otherwise, compute the remainder
    # leftover will be the size of the first chunk
    leftover = T % d

    boundaries = [0]

    # If leftover is nonzero, the first chunk is `leftover` frames
    # from index 0 up to leftover-1
    if leftover != 0:
        boundaries.append(leftover)

    current = leftover
    # Build boundaries in steps of size d
    while current < T:
        current += d
        # Clamp if we go beyond the last index
        if current > T:
            current = T
        boundaries.append(current)

    return boundaries


def uniform_segment_variant(features, d=32):
    """
    Segment `features` into chunks of size d, placing any leftover frames at the end.

    - Chunks of size d are created first.
    - If T % d != 0, the leftover forms the final smaller chunk.

    :param features: A tensor or array of shape (T, D), where T is the temporal length.
    :param d: The fixed chunk size.
    :return: A list of boundary indices (the ends of each chunk).
    """
    T = features.shape[0]
    boundaries = [0]

    current = 0
    while current + d <= T:
        current += d
        boundaries.append(current)

    # If there is leftover at the end
    if current < T:
        boundaries.append(T)

    return boundaries
def cal_left_depth_score(sim_scores):
    n = sim_scores.shape[0]
    depth_scores = torch.zeros(sim_scores.size(), dtype=sim_scores.dtype, device=sim_scores.device)
    # clip = min(max(n//10, 2), 5) # adopt clip to improve efficiency
    for i in range(n):
        lpeak = sim_scores[i]
        for li in range(i-1, -1, -1):
            if sim_scores[li] >= lpeak:
                lpeak = sim_scores[li]
            else:
                break
        depth_scores[i] = lpeak - sim_scores[i]
    return depth_scores


def segment_left(features, alpha=0.5, k=None):
    # input shape: t, d
    sim_scores = torch.cosine_similarity(features[:-1, :], features[1:, :])
    depth_scores = cal_left_depth_score(sim_scores)

    # print(depth_scores)

    if k is not None:
        # select by top k
        boundaries = torch.topk(depth_scores, k).indices.sort()[0]
    else:
        # select by threshold (original)
        std, mean = torch.std_mean(depth_scores)
        thresh = mean + alpha * std
        condition = depth_scores > thresh
        boundaries = condition.nonzero().squeeze(-1)
        # if len(boundaries) > 15: #limit max segments to prevent from OOM: 7 comes from RMT paper
        #     boundaries = torch.topk(depth_scores, 15).indices.sort()[0]

    boundaries = boundaries.tolist()

    # print("boudaries: ", boundaries)
    # print("features: ", features)

    # if type(boundaries) == int or boundaries == [] or boundaries[-1] != features.shape[0]-1:
    #     boundaries.append(features.shape[0]-1)
    if type(boundaries) == int or boundaries == []:
        boundaries.append(features.shape[0]-1)


    # # average segment
    # l = features.shape[0]
    # boundaires = list(range(l//(k+1)-1, l, l//(k+1)))

    # segments = []
    # index = 0
    # for bi in boundaries:
    #     segments.append(features[index: bi+1])
    #     index = bi + 1
    # if index < features.shape[0]: segments.append(features[index:])

    return boundaries


def sample_scenes_priority(features, n=32, alpha=0.5, k=None):
    """
    Sample n frames from features of shape [frames, patches, dim],
    prioritizing surprising scenes if there are too many scenes.
    """
    T = features.shape[0]
    frame_features = features.mean(dim=1)  # flatten spatial dimension

    # segment with your provided function
    # note we capture depth scores to prioritize
    scene_boundaries, depth_scores = segment(frame_features, alpha=alpha, k=k)


    # always include first and last
    if 0 not in scene_boundaries:
        scene_boundaries = [0] + scene_boundaries
    if T not in scene_boundaries:
        scene_boundaries.append(T)
    scene_boundaries = sorted(set(scene_boundaries))
    print("scene_boundaries: ", scene_boundaries)

    # number of scenes
    num_scenes = len(scene_boundaries) - 1

    # if scenes <= n, allocate normally
    if num_scenes <= n:
        frame_budget = [1] * num_scenes
        remaining = n - num_scenes
        scene_lengths = [scene_boundaries[i + 1] - scene_boundaries[i] for i in range(num_scenes)]
        total_len = sum(scene_lengths)
        for i in range(num_scenes):
            frame_budget[i] += int(remaining * scene_lengths[i] / total_len)
        # fix rounding mismatch
        while sum(frame_budget) < n:
            frame_budget[sum(frame_budget) % num_scenes] += 1
        while sum(frame_budget) > n:
            frame_budget[frame_budget.index(max(frame_budget))] -= 1
        # sample
        sampled_indices = []
        for i in range(num_scenes):
            start = scene_boundaries[i]
            end = scene_boundaries[i + 1]
            length = end - start
            k = frame_budget[i]
            if length <= k:
                indices = list(range(start, end))
            else:
                indices = torch.linspace(start, end - 1, steps=k).round().long().tolist()
            sampled_indices.extend(indices)
        return sorted(set(sampled_indices))

    else:
        # too many scenes for n, pick most surprising scenes
        # get the scores for the boundaries, map to scenes
        # boundary i separates scene i and scene i+1, so
        boundary_scores = []
        for b in scene_boundaries[1:-1]:
            boundary_scores.append(depth_scores[b - 1].item())  # note boundary is after b-1
        # assign these scores to scenes
        scene_scores = [0] + boundary_scores  # first scene gets 0
        scored_scenes = list(enumerate(scene_scores))
        # sort scenes by score descending
        top_scenes = sorted(scored_scenes, key=lambda x: -x[1])[:n]
        chosen_scenes = [x[0] for x in top_scenes]
        # sample center frame of each chosen scene
        sampled_indices = []
        for i in chosen_scenes:
            start = scene_boundaries[i]
            end = scene_boundaries[i + 1]
            center = (start + end) // 2
            sampled_indices.append(center)
        return sorted(sampled_indices)
