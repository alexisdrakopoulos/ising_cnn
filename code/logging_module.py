import json
from datetime import datetime

def create_log(log_type, info):

	if log_type not in ("model", "evaluation", "backup"):
		raise ValueError("log_type must be model, evaluation or backup str object")

	time = str(datetime.now())

	log_dict = {time:info}