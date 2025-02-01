test:
	$(POETRY_RUN) python -m pytest -l tests

# run specific tests by filtering file or test name via "-k test_filter"
# example: "make test-k-index_deletion" filters test name by "index_deletion"
test-k-%:
	$(POETRY_RUN) python -m pytest -lvvsx -k $* tests

# input: make test-v
# result: pipenv run python -m pytest -lv tests
test-%:
	$(POETRY_RUN) python -m pytest -l$* tests

test-failed:
	$(POETRY_RUN) python -m pytest -l --last-failed tests
