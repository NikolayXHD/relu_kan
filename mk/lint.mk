# Every other mypy run leads to a ton of false positive warns about ruamel.
# Clearing mypy cache helps.
lint: ruff mypy

clear:
	rm -rf ./.mypy_cache ./.pytest_cache
	find . -type f -name "*.pyc" -delete
	find . -type d -name '.ipynb_checkpoints' -exec rm -r {} \;
	find . -type d -name '__pycache__' -exec rm -r {} \;


# minus to ignore exit code and proceed to next targets
ruff:
	-shopt -s globstar; $(POETRY_RUN) ruff check {src,tests,notebooks}/**/*.py

ruff-fix:
	-shopt -s globstar; $(POETRY_RUN) ruff --fix {src,tests,notebooks}/**/*.py

mypy:
	-$(POETRY_RUN) mypy src tests notebooks \
--namespace-packages --explicit-package-bases --check-untyped-defs
