# CLAUDE.md

## Project Overview

Personal technical blog hosted on GitHub Pages.

## Tech Stack

- **Static site generator**: Jekyll with GitHub Pages (`github-pages` gem)
- **Theme**: [So Simple](https://github.com/mmistakes/so-simple-theme) v3.2.0 (remote theme)
- **Math rendering**: MathJax (tex-mml-chtml combo, AMS tags)
- **Markdown engine**: kramdown with Rouge syntax highlighting
- **Pre-commit**: pre-commit hooks (large file check, trailing whitespace, line endings, etc.)

## Project Structure

```
_posts/          # Blog posts (Markdown with YAML front matter)
_layouts/        # HTML layout templates
_includes/       # Reusable HTML partials
_data/           # Site data (navigation, authors, text)
_sass/           # Stylesheets
_site/           # Generated site (do not edit, gitignored)
assets/          # CSS and JS assets
images/          # Post images
_config.yml      # Jekyll site configuration
```

## Branching & Publishing

- **`pages`** — Publishing branch. GitHub Pages deploys from this branch; pushes here trigger deployment.
- **Other branches** — Working branches. Pushes only trigger CI build validation, not deployment.
- To publish: run `bash publish.sh` to merge the current branch into `pages` and push.

## Common Commands

```bash
# Local development server (port 4001)
bash run.sh

# Publish current branch to pages (triggers deployment)
bash publish.sh

# Install dependencies
bundle install
```

## Writing Posts

- Posts go in `_posts/` with filename format: `YYYY-MM-DD-Title.md`
- Posts use `layout: post` by default (configured in `_config.yml` front matter defaults)
- Math: use `$...$` for inline and `$$...$$` for display math (MathJax enabled)
- Images: place in `images/`, reference as `/images/filename.png`
- Code blocks: use fenced code blocks with language identifier; Rouge provides syntax highlighting with line numbers
