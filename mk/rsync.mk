HOSTS=kolia@dell kolia@i7 kolia@yc kolia@kde kolia@home nmidalgodias@mac

REPO_DIR = $(shell realpath .)
REPO_NAME = $(shell basename $(REPO_DIR))
STORAGE_NAME = $(REPO_NAME).storage
STORAGE_DIR = $(shell realpath ../$(STORAGE_NAME))
REMOTE_STORAGE_DIR = "git/$(STORAGE_NAME)"
REMOTE_REPO_DIR = "git/$(REPO_NAME)"

check-dirs:
	echo REPO_DIR $(REPO_DIR)
	echo REPO_NAME $(REPO_NAME)
	echo STORAGE_NAME $(STORAGE_NAME)
	echo STORAGE_DIR $(STORAGE_DIR)
	echo REMOTE_STORAGE_DIR $(REMOTE_STORAGE_DIR)
	echo REMOTE_STORAGE_DIR $(REMOTE_STORAGE_DIR)
	echo REMOTE_REPO_DIR $(REMOTE_REPO_DIR)
	echo SHM_PREFIX $(SHM_PREFIX)


$(patsubst %,rsync-cache-to-%,$(HOSTS)):
	@$(eval HOST := $(lastword $(subst -, ,$@)))
	rsync -avrR \
--include "*/" --include "*" \
$(STORAGE_DIR)/./cache \
$(HOST):$(REMOTE_STORAGE_DIR)/

$(patsubst %,rsync-cache-from-%,$(HOSTS)):
	@$(eval HOST := $(lastword $(subst -, ,$@)))
	rsync --timeout=10 -avrRm \
--include "*/" --include "*" \
$(HOST):$(REMOTE_STORAGE_DIR)/./cache \
$(STORAGE_DIR)/


$(patsubst %,rsync-repo-to-%,$(HOSTS)):
	@$(eval HOST := $(lastword $(subst -, ,$@)))
	rsync -avrR \
--exclude ".git/" \
--exclude ".mypy_cache/" --exclude ".pytest_cache/" --exclude ".ruff_cache/" \
--exclude ".ipynb_checkpoints/" --exclude ".virtual_documents/" \
--exclude ".lsp_symlink" \
--exclude ".env" \
--include "*/" --include "*" \
$(REPO_DIR)/./ \
$(HOST):$(REMOTE_REPO_DIR)/

$(patsubst %,rsync-repo-from-%,$(HOSTS)):
	@$(eval HOST := $(lastword $(subst -, ,$@)))
	rsync --timeout=10 -avrRm \
--exclude ".git/" \
--exclude ".mypy_cache/" --exclude ".pytest_cache/" --exclude ".ruff_cache/" \
--exclude ".ipynb_checkpoints/" --exclude ".virtual_documents/" \
--exclude ".lsp_symlink" \
--exclude ".env" \
--include "*/" --include "*" \
$(HOST):$(REMOTE_REPO_DIR)/./ \
$(REPO_DIR)/


$(patsubst %,rsync-logs-from-%,$(HOSTS)):
	@$(eval HOST := $(lastword $(subst -, ,$@)))
	mkdir -p $(REPO_DIR)/../sandbox-prod.storage/logs
	rsync -avrR \
$(HOST):git/sandbox-prod.storage/logs/./ \
$(REPO_DIR)/../sandbox-prod.storage/logs/
