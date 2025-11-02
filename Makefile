.PHONY: lint format typecheck
PY_SRC := dsbase tests
PS := powershell -NoProfile -ExecutionPolicy Bypass -Command
.SILENT:

lint:
	$(PS) "Write-Host '==== LINT START ====' -ForegroundColor Yellow"

	@poetry run black --check -q $(PY_SRC) \
	  && $(PS) "Write-Host 'black: PASSED' -ForegroundColor Green" \
	  || ( $(PS) "Write-Host 'black: FAILED (Run: make format)' -ForegroundColor Red"; exit 1 )

	@poetry run isort --check-only -q $(PY_SRC) \
	  && $(PS) "Write-Host 'isort: PASSED' -ForegroundColor Green" \
	  || ( $(PS) "Write-Host 'isort: FAILED (Run: make format)' -ForegroundColor Red"; exit 1 )

	@poetry run flake8 $(PY_SRC) >NUL 2>&1 \
	  && $(PS) "Write-Host 'flake8: PASSED' -ForegroundColor Green" \
	  || ( poetry run flake8 $(PY_SRC) & $(PS) "Write-Host 'flake8: FAILED' -ForegroundColor Red"; exit 1 )

	@poetry run mypy $(PY_SRC) --pretty --show-error-codes \
	  && $(PS) "Write-Host 'mypy: PASSED' -ForegroundColor Green" \
	  || ( $(PS) "Write-Host 'mypy: FAILED' -ForegroundColor Red"; exit 1 )

	$(PS) "Write-Host '==== ALL CHECKS PASSED ====' -ForegroundColor Green"
