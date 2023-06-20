# Variables
SRC_DIR = .
SRC_FILES = $(SRC_DIR)/privateGPT.py $(SRC_DIR)/constants.py
FLAKE8_OPTIONS = --ignore=E501

.PHONY: lint
lint:
	@flake8 $(FLAKE8_OPTIONS) $(SRC_FILES)

.PHONY: format
format:
	@black $(SRC_DIR)

.PHONY: check
check: lint format

.PHONY: help
help:
	@echo "Available targets:"
	@echo "  lint     - Run Flake8 to lint the source code."
	@echo "  format   - Run Black to format the source code."
	@echo "  check    - Run both lint and format."
	@echo "  help     - Show this help message."

# Set the default target to 'help'
.DEFAULT_GOAL := help
