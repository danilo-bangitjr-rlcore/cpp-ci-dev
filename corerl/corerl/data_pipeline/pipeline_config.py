from __future__ import annotations

import logging

from corerl.data_pipeline.transforms import register_dispatchers

logger = logging.getLogger(__name__)
register_dispatchers()
