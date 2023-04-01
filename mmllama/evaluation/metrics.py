from typing import Dict, List, Optional, Sequence, Union

from mmengine.evaluator import BaseMetric

from mmllama.registry import METRICS


@METRICS.register_module()
class DummyMetric(BaseMetric):
    """
    TODO: implement a metric
    """
    default_prefix: Optional[str] = 'metric'

    def __init__(self,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)

    # TODO: data_batch is no longer needed, consider adjusting the
    #  parameter position
    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of data samples that
                contain annotations and predictions.
        """
        for data_sample in data_samples:
            result = dict()
            self.results.append(result)

    def compute_metrics(self, results: list) -> Dict[str, float]:

        return {}
