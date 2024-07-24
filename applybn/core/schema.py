from typing import TypedDict, Optional, Sequence, Tuple, List, Union, Callable


class StructureLearnerParams(TypedDict, total=False):
    init_edges: Optional[Sequence[str]]
    init_nodes: Optional[List[str]]
    remove_init_edges: bool
    white_list: Optional[Tuple[str, str]]
    bl_add: Optional[List[str]]


class ParamDict(TypedDict, total=False):
    scoring_function: Union[Tuple[str, Callable], Tuple[str]]
    progress_bar: bool
    classifier: Optional[object]
    regressor: Optional[object]
    params: Optional[StructureLearnerParams]
    optimizer: str
