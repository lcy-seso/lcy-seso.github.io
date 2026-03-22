#!/bin/bash
set -e

# Publish the current branch to the pages branch (triggers GitHub Pages deployment).
# Usage: bash publish.sh

CURRENT_BRANCH=$(git symbolic-ref --short HEAD)
PUBLISH_BRANCH="pages"

if [ "$CURRENT_BRANCH" = "$PUBLISH_BRANCH" ]; then
  echo "Already on $PUBLISH_BRANCH. Switch to a working branch first."
  exit 1
fi

echo "Publishing $CURRENT_BRANCH -> $PUBLISH_BRANCH"

git checkout "$PUBLISH_BRANCH"
git merge "$CURRENT_BRANCH" --no-edit
git push origin "$PUBLISH_BRANCH"
git checkout "$CURRENT_BRANCH"

echo "Done. Site will be deployed from $PUBLISH_BRANCH."
