from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import partial
from typing import Dict, List, Optional

from tqdm import tqdm

from ...type import MultimodalSample
from ..utils import save_samples


@dataclass
class BasePostProcessorConfig:
    type: str
    name: Optional[str] = None
    args: Dict = field(default_factory=dict)

    def __post_init__(self):
        if self.name is None:
            self.name = self.type


class BasePostProcessor(ABC):
    name: str

    # Opt-in flag: only processors that are cheap to pickle and hold no large
    # in-memory models (e.g. the chunker) should set this True. Model-heavy
    # processors (tagger, NER) keep it False to avoid re-pickling weights per
    # worker. See batch_process.
    parallelizable: bool = False

    def __init__(self, name: str):
        self.name = name

    def __repr__(self):
        return f"{self.__class__.__name__}({self.name})"

    def __call__(
        self, sample: MultimodalSample, **kwargs
    ) -> MultimodalSample | List[MultimodalSample]:
        return self.process(sample, **kwargs)

    @abstractmethod
    def process(self, sample: MultimodalSample, **kwargs) -> List[MultimodalSample]:
        """Abstract method for processing a sample.

        Args:
            sample (MultimodalSample): The sample to process.

        Returns:
            List[MultimodalSample]: The processed sample(s).
        """
        pass

    def batch_process(
        self,
        samples: List[MultimodalSample],
        pool=None,
        tmp_save_path: Optional[str] = None,
        save_every: int = 100,
        **kwargs,
    ) -> List[MultimodalSample]:
        """
        Process a batch of samples.
        Args:
            samples: a list of samples to process
            pool: optional worker pool; when provided and this processor is
                `parallelizable`, samples are processed in parallel (order
                preserved). Ignored otherwise.
            tmp_save_path: if provided, intermediate results will be saved to this path every 100 samples
            kwargs: additional arguments to pass to the process method

        Returns: a list of processed samples
        """
        if tmp_save_path:
            # Clear the file if it exists
            open(tmp_save_path, "w").close()

        if self.parallelizable and pool is not None and len(samples) > 1:
            return self._batch_process_parallel(
                samples, pool, tmp_save_path, save_every, **kwargs
            )

        res = []
        current_batch = []
        for s in tqdm(samples, desc=f"{self.name}"):
            new = self.process(s, **kwargs)
            current_batch += new

            if len(current_batch) >= save_every:
                if tmp_save_path:
                    save_samples(current_batch, tmp_save_path, append_mode=True)
                res += current_batch
                current_batch = []

        if current_batch:
            if tmp_save_path:
                save_samples(current_batch, tmp_save_path, append_mode=True)

            res += current_batch

        return res

    def _batch_process_parallel(
        self,
        samples: List[MultimodalSample],
        pool,
        tmp_save_path: Optional[str],
        save_every: int,
        **kwargs,
    ) -> List[MultimodalSample]:
        # process returns a list per sample; map preserves input order.
        nested = pool.map(partial(self.process, **kwargs), samples)
        res = [s for sub in nested for s in sub]

        if tmp_save_path:
            for i in range(0, len(res), save_every):
                save_samples(res[i : i + save_every], tmp_save_path, append_mode=True)

        return res
