from src.utils.seed import set_seed
from src.utils.device import get_device
from src.utils.logger import logger

set_seed(42)

device = get_device()

logger.info("Project infrastructure initialized successfully.")