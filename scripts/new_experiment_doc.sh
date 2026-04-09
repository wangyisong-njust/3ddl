#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 3 ]; then
  echo "Usage: $0 <template-name> <output-relative-path> <title>"
  echo "Example: $0 experiment optimization_history/100_my_test.md \"My Test\""
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

TEMPLATE_NAME="$1"
OUTPUT_REL="$2"
TITLE="$3"

case "$TEMPLATE_NAME" in
  experiment) TEMPLATE_FILE="$ROOT_DIR/docs/templates/experiment_template.md" ;;
  benchmark) TEMPLATE_FILE="$ROOT_DIR/docs/templates/benchmark_template.md" ;;
  optimization|optimization_note) TEMPLATE_FILE="$ROOT_DIR/docs/templates/optimization_note_template.md" ;;
  *)
    echo "Unknown template: $TEMPLATE_NAME"
    exit 1
    ;;
esac

OUTPUT_FILE="$ROOT_DIR/docs/$OUTPUT_REL"
OUTPUT_DIR="$(dirname "$OUTPUT_FILE")"

if [ -e "$OUTPUT_FILE" ]; then
  echo "Refusing to overwrite existing file: $OUTPUT_FILE"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
DATE_STR="$(date +%F)"

sed \
  -e "s/{{DATE}}/${DATE_STR}/g" \
  -e "s/{{TITLE}}/${TITLE}/g" \
  "$TEMPLATE_FILE" > "$OUTPUT_FILE"

echo "Created: $OUTPUT_FILE"
