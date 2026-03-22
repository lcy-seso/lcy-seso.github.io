#!/bin/bash

bundle exec jekyll serve --port 4001 --drafts 2>&1 | tee jekyll.log
