password:
	$(POETRY_RUN) jupyter server password

.lsp_symlink:
	ln -sf / .lsp_symlink

lab: jupyterlab
jupyterlab: .lsp_symlink
	-$(POETRY_RUN) jupyter lab --no-browser --port 8890 \
--ContentsManager.allow_hidden=True

# this removes .lsp_symlink after jupyterlab task is completed
.INTERMEDIATE: .lsp_symlink

format: jupytext-py format-py jupytext-ipynb

format-py:
	$(POETRY_RUN) python -m black src tests notebooks

jupytext-ipynb:
	$(POETRY_RUN) find notebooks -name '*.py' \
-not -path '*/.ipynb_checkpoints/*' \
| $(call filter_existing_notebook) \
| $(POETRY_RUN) xargs jupytext --sync

jupytext-py:
	$(POETRY_RUN) find notebooks -name '*.ipynb' \
-not -path '*/.ipynb_checkpoints/*' -print0 \
| $(POETRY_RUN) xargs -r0 jupytext --sync

trust-all-notebooks:
	$(POETRY_RUN) find notebooks -name '*.ipynb' \
-not -path '*/.ipynb_checkpoints/*' -exec jupyter trust {} +

# interprets stdin as line-separated list of *.py file paths
# writes to stdout those which have corresponding *.ipynb nearby
# "$${line%.py}" is filename with .py truncated from the end
define filter_existing_notebook
  while IFS= read -r line; do \
    if [ -f "$${line%.py}".ipynb ]; then \
      echo "$$line"; \
    fi; \
  done
endef
