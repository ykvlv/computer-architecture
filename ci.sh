#!/bin/bash

echo '===[STATIC CHECK]==='
echo '------[FLAKE8]------'
flake8 .
echo '------[PYLINT]------'
pylint src tests
echo '-------[MYPY]-------'
mypy .

echo '=======[TEST]======='
echo '----[UNIT-TESTS]----'
pytest --cov=src
