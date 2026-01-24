class SampleSlice:
    fov_ids: set[int]

    def __init__(self, fov_ids: set[int]) -> None:
        self.fov_ids = fov_ids
