class SampleSlice:
    slice_id: str
    fov_ids: set[int]

    def __init__(self, slice_id: str, fov_ids: set[int]) -> None:
        self.slice_id = slice_id
        self.fov_ids = fov_ids
