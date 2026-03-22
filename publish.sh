#!/bin/bash
set -e

# Publish: push master to origin (triggers GitHub Pages deployment).
# Usage: bash publish.sh

git push origin master

echo "Done. Site will be deployed from master."
