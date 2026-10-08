#!/bin/zsh
set -eu
cd "${0:A:h}"
export MPLCONFIGDIR="$PWD/.runtime/matplotlib"
mkdir -p "$MPLCONFIGDIR"
if [[ ! -x .venv/bin/python ]]; then
  print 'Окружение отсутствует. Выполните инструкции в README.md.'
  read -r '?Нажмите Enter для выхода'
  exit 1
fi
exec .venv/bin/python launch.py
