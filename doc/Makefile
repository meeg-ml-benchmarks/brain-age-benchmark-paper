PYTHON?=python

BASEDIR=$(CURDIR)
OUTPUTDIR=$(BASEDIR)/rendered

GITHUB_PAGES_BRANCH=gh-pages

.PHONY: clean render publish

all: render

clean:
	rm -rf build

render:
	python main.py

publish:
	touch $(OUTPUTDIR)/.nojekyll
	ghp-import -m "Generate site" -b $(GITHUB_PAGES_BRANCH) $(OUTPUTDIR)
	git push origin $(GITHUB_PAGES_BRANCH)
