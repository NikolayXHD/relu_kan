include mk/env.mk
include mk/jupyter.mk
include mk/lint.mk
include mk/profiling.mk
include mk/rsync.mk
include mk/test.mk

shared-memory-check:
	find /dev/shm/ -name 'sandbox-nb\\*'

shared-memory-clear:
	find /dev/shm/ -name 'sandbox-nb\\*' -delete

clear-cache:
	find $(STORAGE_DIR)/cache -type d -name joblib -print0 | xargs -r0 rm -r

SHELL := $(shell which bash)

POETRY_RUN = poetry run -q dotenv run
