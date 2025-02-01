env-create:
	poetry lock --no-update && poetry install --no-root

env-update:
	poetry lock --no-update && poetry update

env-remove:
	poetry env remove --all

env-delete: env-remove

env-patch:
	cp template/.env .env
	sed -i 's/$${PWD}/$(subst /,\/,${PWD})/g' .env
