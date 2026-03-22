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
_posts/          # Published blog posts (Markdown with YAML front matter)
_drafts/         # Work-in-progress notes (not published)
_layouts/        # HTML layout templates
_includes/       # Reusable HTML partials
_data/           # Site data (navigation, authors, text)
_sass/           # Stylesheets
_site/           # Generated site (do not edit, gitignored)
assets/          # CSS and JS assets
images/          # Post images
_config.yml      # Jekyll site configuration
```

## Workflow

- **`master`** is the publishing branch. GitHub Pages deploys directly from `master`.
- **`_drafts/`** for work-in-progress notes; **`_posts/`** for published posts.
- To hold back a published post for editing: move it from `_posts/` to `_drafts/`, push to update the live site.
- **Development**: work on a feature branch (e.g., `draft/<topic>`), commit freely, then use `/create-pr` to open a PR to `master`. Always **squash merge** the PR to keep master history clean.

## Common Commands

```bash
# Install dependencies
bundle install

# Local preview (includes drafts, port 4001)
bundle exec jekyll serve --port 4001 --drafts
```

## Writing Posts

- Posts go in `_posts/` with filename format: `YYYY-MM-DD-Title.md`
- Drafts go in `_drafts/` with no date prefix needed
- Posts use `layout: post` by default (configured in `_config.yml` front matter defaults)
- Math: use `$...$` for inline and `$$...$$` for display math (MathJax enabled)
- Images: place in `images/`, reference as `/images/filename.png`
- Code blocks: use fenced code blocks with language identifier; Rouge provides syntax highlighting with line numbers
