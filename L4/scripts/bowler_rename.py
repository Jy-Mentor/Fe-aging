import logging
logger = logging.getLogger(__name__)

from bowler import Query

Query().select_function("train_sage").rename("train_sage_v2").execute()
